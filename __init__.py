import itertools
from pathlib import Path

import numpy as np

from pcigale.data import SimpleDatabase
from pcigale.utils.console import console
from pcigale.utils.counter import Counter


def build(agemax=13700):
    console.rule("Bowling et al. (2026) SSPs w/ Temperature-Dependent IMF")

    path = Path(__file__).parent / "data"

    db = SimpleDatabase("sspt", writable=True)

    agemax = int(agemax)

    ### naming schemes
    metallicity = {
	    'm0004': 0.0004,
        'm0012': 0.0012,
        'm0080': 0.0080,
        'm0200': 0.0200,
        'm0500': 0.0500
    }

    temperature = {
        f"T{T}": T
        for T in range(20, 81)
    }


    ### available models
    available_models = {
        (key, imf): f"{key}_{imf}_ssp.npz"
        for key in metallicity
        for imf in temperature
    }

    ### there is only 1 IMF, but the temperatures changes,
    ### so iteration needs to mutliple models here,
    ### just based on temperature + metallicity here
    combinations = list(
        itertools.product(
            metallicity,
            temperature
        )
    )

    counter = Counter(len(combinations))

    for key, imf in combinations:

        ### skip combinations that do not exist yet
        ### should not happen, everything should be in order
        ### keep here just in case
        if (key, imf) not in available_models:
            counter.inc()
            continue

        filename = path / available_models[(key, imf)]

        ssp = np.load(filename)

        time_grid = ssp["t"]
        wavelength = ssp["wl"]
        info = ssp["info"]
        spectra = ssp["spec"]

        ### limit SSP age range if necessary
        ### shouldn't be, but keep as a guardrail anyway
        age_mask = time_grid <= agemax

        time_grid = time_grid[age_mask]
        info = info[:, age_mask]
        spectra = spectra[:, age_mask]

        db.add(
            {"imf": imf, "Z": metallicity[key]},
            {
                "t": time_grid,
                "wl": wavelength,
                "info": info,
                "spec": spectra,
            },
        )

        counter.inc()

    db.close()
    counter.progress.join()
