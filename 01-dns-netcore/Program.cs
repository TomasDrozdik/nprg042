using System;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Collections.Generic;

namespace dns_netcore
{
	/// <summary>
	/// Representing result of one tested recursive query (and the time it took to process it).
	/// </summary>
	struct TestResult
	{
		public readonly string domain;
		public readonly IP4Addr address;
		public readonly long elapsedMilliseconds;

		public TestResult(string domain, IP4Addr address, long elapsedMilliseconds)
		{
			this.domain = domain;
			this.address = address;
			this.elapsedMilliseconds = elapsedMilliseconds;
		}
	}

	class Program
	{
		/// <summary>
		/// Run some simple recursive queries (as many as we have logical cores) to warm up the thread pool.
		/// This should ensure the pool have sufficient treads warm and ready afterwards.
		/// </summary>
		static void WarmUpThreadPool()
		{
			var tasks = new Task[Environment.ProcessorCount];
			for (int i = 0; i < tasks.Length; ++i) {
				tasks[i] = Task.Delay(50);
			}
			Task.WaitAll(tasks);
		}

		/// <summary>
		/// Start measuring task that executes query and measures its latency.
		/// </summary>
		/// <param name="resolver">Resolver implementation being tested</param>
		/// <param name="domain">Domain to be resolved</param>
		/// <returns>Task which yields TestResult representing this test</returns>
		static Task<TestResult> MeasureQuery(IRecursiveResolver resolver, string domain)
		{
			Stopwatch stopwatch = Stopwatch.StartNew();
			var t = resolver.ResolveRecursive(domain);
			return t.ContinueWith<TestResult>(t => {
				stopwatch.Stop();
				return new TestResult(domain, t.Result, stopwatch.ElapsedMilliseconds);
			});
		}

		/// <summary>
		/// Run a batch of queries simultaneously and wait for them all to finish.
		/// </summary>
		/// <param name="resolver">Resolver implementation being tested</param>
		/// <param name="domains">Array of domains to be resolved</param>
		/// <returns>Sum of measured times (in milliseconds)</returns>
		static long RunTestBatch(IRecursiveResolver resolver, string[] domains)
		{
			Console.Write("Starting ... ");
			var tests = domains.Select(domain => MeasureQuery(resolver, domain)).ToArray();
			Console.WriteLine("{0} tests", tests.Length);
			Task.WaitAll(tests);

			long sum = 0;
			foreach (var test in tests) {
				Console.WriteLine("Domain {0} has IP {1} (elapsed time {2} ms) ",
					test.Result.domain, test.Result.address, test.Result.elapsedMilliseconds);
				sum += test.Result.elapsedMilliseconds;
			}
			if (tests.Length > 0) {
				Console.WriteLine("Avg delay {0} ms", sum / tests.Length);
			}
			return sum;
		}

		static long RunTestBatchNoPrintOut(IRecursiveResolver resolver, string[] domains)
		{
			var tests = domains.Select(domain => MeasureQuery(resolver, domain)).ToArray();
			Task.WaitAll(tests);

			long sum = 0;
			foreach (var test in tests) {
				sum += test.Result.elapsedMilliseconds;
			}
			if (tests.Length > 0) {
			}
			return sum;
		}

		static private void ComparisonTestBatch(DNSClient client, string[] domains) {

			var resolver = new RecursiveResolver(client);
			for(int i = 0; i < 1; ++i) {
				var sum = RunTestBatch(resolver, domains);
				Console.WriteLine("TIME: {0} ms", sum);
			}
			/*var serialResolver = new SerialRecursiveResolver(client);
			var resolver = new RecursiveResolver(client);
			var sumSerial = RunTestBatch(serialResolver, domains);
			serialResolver = new SerialRecursiveResolver(client);
			var sum = RunTestBatch(serialResolver, domains);
			serialResolver = new SerialRecursiveResolver(client);
			var sumSerial0 = RunTestBatch(serialResolver, domains);
			serialResolver = new SerialRecursiveResolver(client);
			var sum0 = RunTestBatch(serialResolver, domains);
			serialResolver = new SerialRecursiveResolver(client);
			var sum1 = RunTestBatch(serialResolver, domains);
			serialResolver = new SerialRecursiveResolver(client);
			var sum2 = RunTestBatch(serialResolver, domains);
			Console.WriteLine("TIME: {0} ms", sumSerial);
			Console.WriteLine("TIME: {0} ms", sum);
			Console.WriteLine("TIME: {0} ms", sumSerial0);
			Console.WriteLine("TIME: {0} ms", sum0);
			Console.WriteLine("TIME: {0} ms", sum1);
			Console.WriteLine("TIME: {0} ms", sum2);
			Console.WriteLine("AVG EFFICIENCY IS {0} %", sumSerial * 100 / sum);
			Console.WriteLine("AVG EFFICIENCY IS {0} %", sumSerial0 * 100 / sum0);*/
		}

		static private StringBuilder GenerateRandomString(Random rnd, int length) {
            StringBuilder builder = new StringBuilder();

            for (int i = 0; i < length; ++i) {
				builder.Append((char)('a' + rnd.Next(26)));
			}
			return builder;
		}

		static private string GenerateRandomDomain(Random rnd, int minLength, int maxLength) {

			int subDomainsCount = 1 + rnd.Next(8);
			StringBuilder builder = new StringBuilder();

			for(int i = 0; i < subDomainsCount; ++i) {
				int length = minLength + rnd.Next(maxLength-minLength);
				builder.Append(GenerateRandomString(rnd, length));
				builder.Append('.');
			}
			builder.Remove(builder.Length-1, 1); // Removes the last dot
			return builder.ToString();
		}

		static private IEnumerable<string> GenerateLevelDomains(Random rnd, int minLevels, int maxLevels, int minLength, int maxLength, int minFork, int maxFork) {

			int level = minLevels + rnd.Next(maxLevels - minLevels);

			int fork = minFork + rnd.Next(maxFork - minFork);
			IEnumerable<string> domains = new List<string>();
			for (int i = 0; i < fork; ++i) {
				int length = minLength + rnd.Next(maxLength - minLength);
				var sub = GenerateRandomString(rnd, length).ToString();
				IEnumerable<string> domain;
				if (level <= 0)
					domain = new string[] {sub};
				else
					domain = GenerateLevelDomains(rnd, minLevels-1, maxLevels-1, minLength, maxLength, minFork, maxFork).Select(d => d + "." + sub);
				domains = domains.Concat(domain);
			}
			return domains;
		}

		static private string[] GenerateDomains(Random rnd) {

			int count = 50;
			int minLength = 2;
			int maxLength = 8;
			int minLevels = 1;
			int maxLevels = 5;
			int minFork = 1;
			int maxFork = 2;
			List<string> domains = new List<string>();
			for (int i = 0; i < count; ++i) {
				domains = domains.Concat(GenerateLevelDomains(rnd, minLevels, maxLevels, minLength, maxLength, minFork, maxFork)).ToList();
			}
			return domains.ToArray();
		}

		static private string GetRandomSubDomain(Random rnd, string domain) {
			var subs = domain.Split('.');
			int subCount = rnd.Next( subs.Length - 1 );
			StringBuilder builder = new StringBuilder();
			for (int i = subCount; i < subs.Length; ++i) {
				builder.Append(subs[i]);
				builder.Append('.');
			}
			builder.Remove(builder.Length-1, 1); // Removes the last dot
			return builder.ToString();
		}

		static string[] Shuffle(Random rnd, IList<string> list) {
			string[] shuffledList = new string[list.Count];

			foreach(var item in list) {
				int position = rnd.Next(list.Count);
				for(int i = 0; shuffledList[position] != null; ++i){
					position = rnd.Next(list.Count);
					if (i > 10)
						while(shuffledList[position] != null)
							position = (position+1) % list.Count;
				}
				shuffledList[position] = item;
			}
			return shuffledList;
		}

		static void Main(string[] args)
		{
			Random rnd = new Random(123456789);
			var client = new DNSClient();

			var defaultList = new string[] {
				"www.ksi.ms.mff.cuni.cz",
				"parlab.ms.mff.cuni.cz",
				"www.seznam.cz",
				"www.google.com",
				"www.share.home.alabanda.cz" } ;
			var generatedList = GenerateDomains(rnd);
			var domainList = generatedList;

			client.InitData( domainList );
			Console.WriteLine("Total {0} logical processors detected.", Environment.ProcessorCount);
			Console.WriteLine("Warming up thread pool...");
			WarmUpThreadPool();

			Console.WriteLine("{0}", ThreadPool.ThreadCount);
			string[] testBatch = domainList.Take(100).ToArray();
			ComparisonTestBatch(client, Shuffle(rnd, testBatch));

			Console.WriteLine("{0}", ThreadPool.ThreadCount);
		}
	}
}
