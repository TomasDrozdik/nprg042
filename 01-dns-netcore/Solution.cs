using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Text;
using System.Threading.Tasks;

namespace dns_netcore
{
	class RecursiveResolver : IRecursiveResolver
	{
		private IDNSClient dnsClient;

		public RecursiveResolver(IDNSClient client)
		{
			this.dnsClient = client;
		}

		public Task<IP4Addr> ResolveRecursive(string domain)
		{
			/*
			 * Just copy-pasted code from serial resolver.
			 * Replace it with your implementation.
			 * Also you may change this method to async (it will work with the interface).
			 */

			return Task<IP4Addr>.Run(() => {
				string[] domains = domain.Split('.');
				Array.Reverse(domains);
				IP4Addr res = dnsClient.GetRootServers()[0];
				foreach (var sub in domains) {
					var t = dnsClient.Resolve(res, sub);
					t.Wait();
					res = t.Result;
				}
				return res;
			});
		}
	}
}
