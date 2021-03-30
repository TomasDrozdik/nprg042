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

		private ConcurrentDictionary<IP4Addr, Task<string>> reversers;

		// Atomic counter used to schedule root servers among resolvers. To obtain next root server index modulo this
		// with root server count.
		private int rootServerCounter = 0;

		public RecursiveResolver(IDNSClient client)
		{
			this.dnsClient = client;
			this.resolvers = new ConcurrentDictionary<ParsedDomain, Task<IP4Addr>>();
			this.reversers = new ConcurrentDictionary<IP4Addr, Task<string>>();
		}

		public async Task<IP4Addr> ResolveRecursive(string domain)
		{
			ParsedDomain parsedDomain = new ParsedDomain(domain);
			return await AssignResolverAndResolve(parsedDomain);
		}

		// Look for a resolver currently resolving this domain.
		// If there is one just wait for it and return, if there is none create a new one and wait for that one.
		private async Task<IP4Addr> AssignResolverAndResolve(ParsedDomain domain)
		{
			Task<IP4Addr> resolver;
			var hasResolver = resolvers.TryGetValue(domain, out resolver);
			if (hasResolver) {
				if (resolver.Status == TaskStatus.RanToCompletion) {
					try {
						var reversed = new ParsedDomain(AsignReverserAndReverse(resolver.Result).Result);
						if (domain.Equals(reversed)) {
							// Reverse successful
							return resolver.Result;
						}
					} catch (DNSClientException) {
						// Reverse unsucessful -> cache is out of date DNSClient.Reverse() did throw -> re-resolve
					}
					// Reverse unsucessful -> re-resolve this domain and update resolver.
					resolvers.TryUpdate(domain, Resolve(domain), resolver);
				} else {
					// Wait for the running task.
					return resolver.Result;
				}
			}

			resolver = resolvers.GetOrAdd(domain, Resolve(domain));
			return resolver.Result;
		}

		private async Task<IP4Addr> Resolve(ParsedDomain domain)
		{
			if (domain.IsTopLevelDomain()) {
				int rootServerIndex = Interlocked.Increment(ref rootServerCounter) % dnsClient.GetRootServers().Count;
				IP4Addr rootServerIP = dnsClient.GetRootServers()[rootServerIndex];
				return dnsClient.Resolve(rootServerIP, domain.GetThisLevelSubdomain()).Result;
			} else {
				var upperLevelIP = await AssignResolverAndResolve(domain.GetUpperLevel());
				return dnsClient.Resolve(upperLevelIP, domain.GetThisLevelSubdomain()).Result;
			}
		}

		private async Task<String> AsignReverserAndReverse(IP4Addr ip)
		{
			Task<string> reverser;
			var hasReverser = reversers.TryGetValue(ip, out reverser);
			if (hasReverser) {
				return reverser.Result;
			} else {
				reverser = reversers.GetOrAdd(ip, dnsClient.Reverse(ip));
				return reverser.Result;
			}
		}
	}
}
