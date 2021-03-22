using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;

namespace dns_netcore
{
	class RecursiveResolver : IRecursiveResolver
	{
		// Immutable class used to wrap a domain in a way that prevents expensive operations on strings.
		// It is meant to work as a key to dictionaries in a sense where 2 overlapping domains with common suffix hash
		// to the same value. For example domain1 "mff.cuni.cz" and domain2 "cuni.cz" can have equal representation as
		// parsed domain where:
		// parsedDomain1.domains = [cz, cuni, mff]
		// parsedDomain2.domains = [cz, cuni]
		// if parsedDomain1.index == parsedDomain2.index == 1 or 0
		//     parsedDomain1 == parsedDomain2 && parsedDomain1.GetHashCode() == parsedDomain2.GetHashCode()
		private class ParsedDomain
		{
			public readonly string[] domains;

			// Index specifies part of the domain that this immutable class represents.
			// E.g. for domains [cz, cuni, mff] and index 1 this class represents "cuni.cz" domain.
			public readonly int index;

			public ParsedDomain(string domain)
			{
				this.domains = domain.Split('.');

				// Reverse the array because it makes for simpler Equals, GetHashCode
				Array.Reverse(domains);
				// Set index of the current domain to the most specific i.e. length of the array
				this.index = this.domains.Length - 1;
			}

			private ParsedDomain(string[] domains, int index)
			{
				this.domains = domains;
				this.index = index;
			}

			// Get upper level domain (up to TLD).
			public ParsedDomain GetUpperLevel()
			{
				Debug.Assert(index > 0 && index <= domains.Length - 1);
				return new ParsedDomain(domains, index - 1);
			}

			public string GetThisLevelSubdomain()
			{
				Debug.Assert(index >= 0 && index < domains.Length);
				return domains[index];
			}

			public bool IsTopLevelDomain() => index == 0;

			public override string ToString() => $"ParsedDomain: domains({String.Join('.', domains)}), index({index})";

			// Equals considers domains[0..index] range
            public override bool Equals(object obj)
            {
				Debug.Assert(index >= 0 && index < domains.Length);
				if (!(obj is ParsedDomain)) {
					return false;
				}
				ParsedDomain other = (ParsedDomain) obj;
				if (index != other.index) {
					return false;
				}
				for (int i = index; i >= 0; --i) {
					if (domains[i] != other.domains[i]) {
						return false;
					}
				}
				return true;
            }

			// GetHashCode considers domains[0..index] range
            public override int GetHashCode()
            {
				Debug.Assert(index >= 0 && index < domains.Length);
				int hash = index;
				for (int i = index; i >- 0; --i) {
					hash = HashCode.Combine(domains[i].GetHashCode(), hash);
				}
				return hash;
            }
		}

		private IDNSClient dnsClient;

		// Dictionary that maps currently resolving domains to the running tasks.
		private ConcurrentDictionary<ParsedDomain, Task<IP4Addr>> resolvers;

		// Dictionary that maps already resolved domains to the the results, that may be no loger valid, and need to be
		// check with DNCClient.Reverse method.
		private ConcurrentDictionary<ParsedDomain, IP4Addr> resolvedCache;

		// Atomic counter used to schedule root servers among resolvers. To obtain next root server index modulo this
		// with root server count.
		private int rootServerCounter = 0;

		public RecursiveResolver(IDNSClient client)
		{
			this.dnsClient = client;
			this.resolvers = new ConcurrentDictionary<ParsedDomain, Task<IP4Addr>>();
			this.resolvedCache = new ConcurrentDictionary<ParsedDomain, IP4Addr>();
		}

		public async Task<IP4Addr> ResolveRecursive(string domain)
		{
			ParsedDomain parsedDomain = new ParsedDomain(domain);
			return await AssignResolverAndResolve(parsedDomain);
		}

		// Look for a resolver currently resolving this domain.
		// If there is one just wait for it and return, if there is none create a new one and wait for that one.
		private async Task<IP4Addr> AssignResolverAndResolve(ParsedDomain parsedDomain)
		{
			// In order to atomically update the resolvers dictionary we have to create a task that would potentially
			// be a new resolver, don't star the task immediately, because if there is a chaced resolver there is no
			// need to start new resolver.
			Task<IP4Addr> potentionalResolver = new Task<IP4Addr>(() => Resolve(parsedDomain).Result);
			Task<IP4Addr> actualResolver = resolvers.GetOrAdd(parsedDomain, potentionalResolver);
			if (actualResolver == potentionalResolver) {
				// GetOrAdd did Add thus there was no cached resovler => start the new task.
				actualResolver.Start();
			}
			return await actualResolver;
		}

		private async Task<IP4Addr> Resolve(ParsedDomain parsedDomain)
		{
			// Check cache.
			IP4Addr cachedIP;
			var hasCachedIP = resolvedCache.TryGetValue(parsedDomain, out cachedIP);
			if (hasCachedIP) {
				var reversedDomain = await dnsClient.Reverse(cachedIP);
				if (reversedDomain == parsedDomain.GetThisLevelSubdomain()) {
					return cachedIP;
				}
				// If the cached value is no loger valid continue and let this resolver resolve it again.
			}

			// Resolve current domain either as TLD or depend on upper level domain.
			IP4Addr ip;
			if (parsedDomain.IsTopLevelDomain()) {
				int rootServerIndex = Interlocked.Increment(ref rootServerCounter) % dnsClient.GetRootServers().Count;
				IP4Addr rootServerIP = dnsClient.GetRootServers()[rootServerIndex];
				ip = await dnsClient.Resolve(rootServerIP, parsedDomain.GetThisLevelSubdomain());
			} else {
				var upperLevelIP = await AssignResolverAndResolve(parsedDomain.GetUpperLevel());
				ip = await dnsClient.Resolve(upperLevelIP, parsedDomain.GetThisLevelSubdomain());
			}

			// Update cache and remove itself from resolvers.
			resolvedCache.AddOrUpdate(parsedDomain, ip, (parsedDomain, oldIP) => oldIP = ip);
			Task<IP4Addr> thisTaskDummy;
			bool removeSuccessful = resolvers.Remove(parsedDomain, out thisTaskDummy);
			Debug.Assert(removeSuccessful);
			return ip;
		}
	}
}
