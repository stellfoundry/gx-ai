import matplotlib
import sys
from load_files import load_files
from extract_species import extract_species
from plot_spectra_1D import plot_spectra_1D


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

    # Input prompts
    spectra       = ["Exit", 'Wkyst', 'Pkyst', 'Qkyst', 'Gamkyst', 'Phi2kyt']
    spectra_label = ["", r"$W_s(k_y)$", r"$[1-\Gamma_0(b_s)]|\phi(k_y)|^2$", r"$Q_s(k_y)$", r"$\Gamma_s(k_y)$", r"$|\phi(k_y)|^2$"]
    choices       = ["Exit", "Plot Wspectra", "Plot Pspectra", "Plot Qspectra", "Plot Gamspectra", "Plot Phi2spectra"]
    input_prompt  = "Please select an option:\n"

    print("")

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
            spectra[choice]
        except:
            print("Invalid choice. Returning to options.")
            print("")
            continue
        if (choice == 0):
            print("")
            break
        try:
            simulations = load_files(filenames, groups=['Inputs', 'Spectra'], spectra=[spectra[choice]])
            plot_spectra_1D(simulations, spectrum=spectra[choice], spectrum_label=spectra_label[choice], \
                            spectrum_range='ky', spectrum_range_label=r"$k_y$", spectrum_xscale="log", spectrum_yscale="log", average_fraction=0.5, absolute_value=False, time_plots=False)
        except KeyboardInterrupt:
            pass

    print("")