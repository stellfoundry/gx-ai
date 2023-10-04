import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle, islice
import sys
from load_files import load_files
from extract_species import extract_species

def plot_fluxes(simulations, flux, flux_label, average_fraction=0.5, norm_vth=np.sqrt(2)):

    print("")
    print("Taking reference species from first simulation provided.")
    print('─' * 100)

    ref_species_list = []

    # Initialising plotting
    fig, (ax1) = plt.subplots(1, 1)
    fig.canvas.manager.set_window_title("Fluxes (all simulations)")
    plot_color = plt.cm.rainbow(np.linspace(0, 1, len(simulations.keys())))

    # Iterating over similation files 
    for simulation_index, simulation_key in enumerate(simulations.keys()):

        # Selecting simulation 
        simulation = simulations[simulation_key]

        # Check whether simulation has same reference species
        species_list, ref_species = extract_species(simulation)
        ref_species_list.append(ref_species)

        if not (ref_species == ref_species_list[0]):
            print("{:} -- WARNING: Inconsistent reference species.".format(simulation_key.name))
            continue

        # Extracting fluxes and averaging
        try:
            flux_plot = simulation['Fluxes'][str(flux)]/norm_vth**3
            t_range   = simulation['Dimensions']['time']
        except:
            print("%s -- WARNING: '%s' not found." %(simulation_key.name, flux))
            continue

        it_start     = int(len(t_range) * average_fraction)
        flux_average = np.average(flux_plot[it_start:, :], axis = 0)
        flux_error   = np.std(flux_plot[it_start:, :], axis = 0)

        # Plotting
        if ref_species == "ions":
            linestyles_dict = {"ions": "solid", "electrons": "dashed"} | dict(zip(species_list[2:], list(islice(cycle(["dotted", "dashdot"]), len(species_list[2:])))))
        else:
            linestyles_dict = {"ions": "dashed", "electrons": "solid"} | dict(zip(species_list[2:], list(islice(cycle(["dotted", "dashdot"]), len(species_list[2:])))))
    
        for species_index, species in enumerate(species_list):
            
            print(r"%s -- %s_%s/%s_gB%s = %.5g +/- %.5g" % (simulation_key.name, flux_label, species[0], flux_label, ref_species[0], flux_average[species_index], flux_error[species_index]))

            if species_index == 0:
                plot_label = simulation_key.name
                # plot_label = simulation['Geometry']['q']
            else:
                plot_label = None

            ax1.plot(t_range, flux_plot[:, species_index], label=plot_label, linewidth=2, linestyle=linestyles_dict[species], color=plot_color[simulation_index])
        
    # Setting plotting options
    ax1.set_xscale("linear")
    ax1.set_xlabel(r"$t(\sqrt{2}a/v_{\mathrm{th}%s})$" % (ref_species[0]))
    ax1.set_yscale("linear")
    ax1.set_ylabel(r"$%s/%s_{\mathrm{gB}%s}$" % (flux_label, flux_label, ref_species[0]))

    # plt.legend(labelcolor='linecolor', handlelength=0, title="Simulations", loc="lower right")
    plt.legend(labelcolor='linecolor', title="Simulations", loc="lower right")
    # plt.legend()

    print("")
    print("Linestyles:", linestyles_dict)

    print('─' * 100)
    print("Completed.")
    print("")

    plt.show()

   
if __name__ == "__main__":

    # Interpreting command line input
    filenames = sys.argv[1:]

    if "latex" in filenames:
        filenames.remove("latex")

        font = {'family' : 'serif',
        'serif'  : ['Computer Modern Roman'],
        'weight' : 'bold',
        'size'   : 18} # Use 40 with latex for papers, 18 otherwise

        matplotlib.rc('font', **font)
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['axes.labelpad']='20'

        print("")
        print("Using LaTeX")

    simulations = load_files(filenames, groups = ['Inputs', 'Fluxes'])

    # Input prompts
    fluxes       = ["Exit", "qflux", "pflux"]
    fluxes_label = ["", "Q", "\Gamma"]
    choices      = ["Exit", "Plot heatflux", "Plot particle flux"]
    input_prompt = "Please select an option:\n"

    for i in range(len(choices)):
        input_prompt += "   {:d}: {:s}\n".format(i, choices[i])

    while (True):
        try:
            choice = int(input(input_prompt))
        except:
            print("Invalid choice. Returning to options.")
            print("")
            continue
        try:
            fluxes[choice]
        except:
            print("Invalid choice. Returning to options.")
            print("")
            continue
        if (choice == 0):
            print("")
            break
        try:
            plot_fluxes(simulations, flux = fluxes[choice], flux_label = fluxes_label[choice], average_fraction = 0.5)
        except KeyboardInterrupt:
            pass

    print("")




