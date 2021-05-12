#define _CRT_SECURE_NO_WARNINGS

/*
 * Main file of the testing framework.
 * Instantiates the program with given data and test its execution.
 */

#include <implementation.hpp>

// internal files
#include <verifier.hpp>
#include <interface.hpp>
#include <serial.hpp>
#include <data.hpp>
#include <stopwatch.hpp>
#include <args.hpp>
#include <exception.hpp>

// libraries
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>



/*
 * Application Entry Point
 */
int main(int argc, char **argv)
{
	typedef double real_t;
	typedef std::uint32_t index_t;
	typedef std::uint32_t length_t;

	bpp::ProgramArguments args(1, 1);
	try {
		args.setNamelessCaption(0, "Path to binary graph file to be processed.");

		// General parameters
		args.registerArg(new bpp::ProgramArguments::ArgBool("verify", "Results of each iteration are verified by a serial algorithm."));
		args.registerArg(new bpp::ProgramArguments::ArgBool("verbose", "Print out debugging messages."));
		args.registerArg(new bpp::ProgramArguments::ArgInt("iterations", "Number of iterations performed.", false, 20, 1));
		args.registerArg(new bpp::ProgramArguments::ArgFloat("verify_tolerance", "Number of iterations performed.", false, 0.001));

		// Physical model parameters.
		args.registerArg(new bpp::ProgramArguments::ArgFloat("vertex_repulsion", "Vertex repulsion force constant.", false, 0.1, 0.00001, 1000000000.0));
		args.registerArg(new bpp::ProgramArguments::ArgFloat("vertex_mass", "Vertex mass constant (affects the vertex momentum).", false, 1.0, 0.00001, 1000000000.0));
		args.registerArg(new bpp::ProgramArguments::ArgFloat("edge_compulsion", "Edge compulsion force constant.", false, 20.0, 0.00001, 1000000000.0));
		args.registerArg(new bpp::ProgramArguments::ArgFloat("slowdown", "Vertex velocity modificator per iteration (1.0 = no slowdown, 0.0 = no motion).", false, 0.995, 0.0, 1.0));
		args.registerArg(new bpp::ProgramArguments::ArgFloat("time_quantum", "Time quantum simulated by each iteration.", false, 0.001, 0.0, 1000000.0));

		args.process(argc, argv);
	}
	catch (bpp::ArgumentException &e) {
		std::cerr << "Argument error: " << e.what() << std::endl;
		args.printUsage(std::cerr);
		return 1;
	}
	bool verbose = args.getArgBool("verbose").getValue();


	try {
		/*
		 * Prepare data and fetch tested program implementation.
		 */
		if (verbose) {
			std::cout << "Loading data from file '" << args[0] << "' ..." << std::endl;
		}

		Graph<real_t, index_t, length_t> graph;
		try {
			graph.load(args[0]);
			if (verbose) {
				std::cout << "Data loaded (" << graph.pointCount() << " points and " << graph.edgeCount() << " edges)." << std::endl;
			}
		}
		catch (DataException &e) {
			std::cerr << "Data Loading Error: " << e.what() << std::endl;
			return 1;
		}


		/*
		 * Prepare program instance, load and compile kernels.
		 */
		if (verbose) {
			std::cout << "Program object preinitialization ... " << std::endl;
		}
		IProgramPotential<real_t, index_t, length_t> *program = new ProgramPotential<real_t, index_t, length_t>;

		// Get model parameters from application arguments.
		ModelParameters<real_t> modelParameters;
		modelParameters.vertexRepulsion = (real_t)args.getArgFloat("vertex_repulsion").getValue();
		modelParameters.vertexMass = (real_t)args.getArgFloat("vertex_mass").getValue();
		modelParameters.edgeCompulsion = (real_t)args.getArgFloat("edge_compulsion").getValue();
		modelParameters.slowdown = (real_t)args.getArgFloat("slowdown").getValue();
		modelParameters.timeQuantum = (real_t)args.getArgFloat("time_quantum").getValue();

		// Set all necessary parameters of the program.
		program->preinitialize(modelParameters, verbose);


		/*
		 * Solution Initialization
		 */
		index_t iterations = (index_t)args.getArgInt("iterations").getValue();

		if (verbose) {
			std::cout << "Solution initialization ... " << std::endl;
		}

		try {
			bpp::Stopwatch stopwatch(true);
			program->initialize(graph.pointCount(), graph.getEdges(), graph.getLengths(), iterations);
			stopwatch.stop();
			if (verbose)
				std::cout << "Elapsed time: " << stopwatch.getMiliseconds() << " ms" << std::endl;
			else
				std::cout << stopwatch.getMiliseconds() << " ";
		}
		catch (UserException &e) {
			std::cerr << "User Exception caught in initialization: " << e.what() << std::endl;
			return 10;
		}


		/*
		 * Run Simulation
		 */
		double totalTime = 0.0;
		bool verify = args.getArgBool("verify").getValue();
		real_t tolerance = (real_t)args.getArgFloat("verify_tolerance").getValue();

		Verifier<real_t, index_t, length_t> *verifier = (verify)
			? (new VerifierFull<real_t, index_t, length_t>(tolerance, verbose, 10))
			: (new Verifier<real_t, index_t, length_t>(tolerance, verbose, 10));
		verifier->init(graph, modelParameters);
		std::vector< Point<real_t> > velocities;

		for (index_t iter = 0; iter < iterations; ++iter) {
			try {
				// Run the solution ...
				bpp::Stopwatch stopwatch(true);
				program->iteration(graph.getPoints());
				stopwatch.stop();
				totalTime += stopwatch.getMiliseconds();

				if (!verify) continue;

				// Perform verificaton ...
				program->getVelocities(velocities);
				if (!verifier->check(graph.getPoints(), velocities)) {
					if (verbose)
						std::cout << "Result verification failed in " << iter << "-th iteration" << std::endl;
					else
						std::cout << "FAILED" << std::endl;
					return 20;
				}
			}
			catch (UserException &e) {
				std::cerr << "User Exception caught in iteration [" << iter << "]: " << e.what() << std::endl;
				return 11;
			}
		}

		// Print total time.
		if (verbose)
			std::cout << "Time per iteration: ";
		std::cout << totalTime / (double)iterations << ((verbose) ? " ms" : "") << std::endl;

		if (verbose)
			verifier->printStats();
		delete verifier;

		return 0;
	}
	catch (std::exception &e) {
		// General exception handler.
		std::cerr << "Gerenal Error: " << e.what() << std::endl;
		return 30;
	}
}
