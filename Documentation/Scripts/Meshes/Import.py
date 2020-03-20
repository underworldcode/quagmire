# Just a quick check to see that all the relevant packages work


# %% define function to import and report "path"

def import_tester(module_name):

    import importlib
    print("Attempting to import module: {}".format(module_name), end="")
    try:
        imported = importlib.import_module(module_name)
        print("\tSuccess - {}".format(imported.__file__))
    except:
        print ("\tFailure !")

# %% Test required imports

import_tester("numpy")
import_tester("scipy")
import_tester("stripy")
import_tester("mpi4py")
import_tester("petsc4py")
import_tester("quagmire")
import_tester("lavavu")

# %%
