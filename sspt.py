### this is the SED module for the Bowling+2026 stellar populations
### computed with a temperature-dependent initial mass function 

import numpy as np

from pcigale.data import SimpleDatabase as Database
from pcigale.sed_modules import SedModule

__category__ = "SSP"


class SSPT(SedModule):

    parameters = {
        "imf": (
            "cigale_list(dtype=int, options=20 & 21 & 22 & 23 & 24 & 25 & 26 & 27 & 28 & 29 & 30 & 31 & 32 & 33 & 34 & 35 & 36 & 37 & 38 & 39 & 40 & 41 & 42 & 43 & 44 & 45 & 46 & 47 & 48 & 49 & 50 & 51 & 52 & 53 & 54 & 55 & 56 & 57 & 58 & 59 & 60 & 61 & 62 & 63 & 64 & 65 & 66 & 67 & 68 & 69 & 70 & 71 & 72 & 73 & 74 & 75 & 76 & 77 & 78 & 79 & 80)",
            "Gas temperature in SF molecular clouds [K]. Every temperature "
            "between 20 K and 80 K in 1 K steps is available.",
            20
        ),
        "metallicity": (
            "cigale_list(options=0.0004 & 0.0012 & 0.0080 & 0.0200 & 0.0500)",
            "Metallicity of the stellar populations. Options are: "
            "0.0004, 0.0012, 0.0080, 0.0200, 0.0500.",
            0.0200
        ),
        "separation_age": (
            "cigale_list(dtype=int, minvalue=0)",
            "Age separating young and old stellar populations [Myr]. "
            "The default value is 10 Myr. "
            "Set to 0 to only model an old stellar population.",
            10
        )
    }

    def convolve(self, sfh):

        info = self.ssp.info[:, :sfh.size]
        spec = self.ssp.spec[:, :sfh.size]

        if self.separation_age == 0:
            # everything is old
            info_young = np.zeros(info.shape[0])

            spec_young = np.zeros(spec.shape[0])

            info_old = 1e6 * np.dot(
                info,
                sfh[::-1]
            )

            spec_old = 1e6 * np.dot(
                spec,
                sfh[::-1]
            )

        else:

            info_young = 1e6 * np.dot(
                info[:, :self.separation_age],
                sfh[-self.separation_age:][::-1]
            )

            spec_young = 1e6 * np.dot(
                spec[:, :self.separation_age],
                sfh[-self.separation_age:][::-1]
            )

            info_old = 1e6 * np.dot(
                info[:, self.separation_age:],
                sfh[:-self.separation_age][::-1]
            )

            spec_old = 1e6 * np.dot(
                spec[:, self.separation_age:],
                sfh[:-self.separation_age][::-1]
            )

        info_all = info_young + info_old

        info_young = dict(
            zip(["m_star", "m_gas", "n_ly"], info_young)
        )

        info_old = dict(
            zip(["m_star", "m_gas", "n_ly"], info_old)
        )

        info_all = dict(
            zip(["m_star", "m_gas", "n_ly"], info_all)
        )

        info_all["age_mass"] = np.average(
            self.ssp.t[:sfh.size],
            weights=info[0, :] * sfh[::-1]
        )

        return (
            spec_young,
            spec_old,
            info_young,
            info_old,
            info_all,
        )

    def _init_code(self):

        self.imf = int(self.parameters["imf"])
        self.Z = float(self.parameters["metallicity"])
        self.separation_age = int(
            self.parameters["separation_age"]
        )

        with Database("sspt") as db:

            self.ssp = db.get(
                imf=f'T{self.imf}',
                Z=self.Z,
            )

    def process(self, sed):

        (
            spec_young,
            spec_old,
            info_young,
            info_old,
            info_all,
        ) = self.convolve(sed.sfh)

        wave = self.ssp.wl

        # Lyman continuum luminosity
        w = np.where(wave <= 91.2)

        lum_lyc_young, lum_lyc_old = np.trapezoid(
            [spec_young[w], spec_old[w]],
            wave[w]
        )

        # total stellar luminosity
        lum_young, lum_old = np.trapezoid(
            [spec_young, spec_old],
            wave
        )

        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.imf", self.imf)
        sed.add_info("stellar.metallicity", self.Z)

        sed.add_info(
            "stellar.old_young_separation_age",
            self.separation_age,
            unit="Myr",
        )

        ### young population
        sed.add_info(
            "stellar.m_star_young",
            info_young["m_star"],
            True,
            unit="solMass",
        )

        sed.add_info(
            "stellar.m_gas_young",
            info_young["m_gas"],
            True,
            unit="solMass",
        )

        sed.add_info(
            "stellar.n_ly_young",
            info_young["n_ly"],
            True,
            unit="ph/s",
        )

        sed.add_info(
            "stellar.lum_ly_young",
            lum_lyc_young,
            True,
            unit="W",
        )

        sed.add_info(
            "stellar.lum_young",
            lum_young,
            True,
            unit="W",
        )

        ### old population
        sed.add_info(
            "stellar.m_star_old",
            info_old["m_star"],
            True,
            unit="solMass",
        )

        sed.add_info(
            "stellar.m_gas_old",
            info_old["m_gas"],
            True,
            unit="solMass",
        )

        sed.add_info(
            "stellar.n_ly_old",
            info_old["n_ly"],
            True,
            unit="ph/s",
        )

        sed.add_info(
            "stellar.lum_ly_old",
            lum_lyc_old,
            True,
            unit="W",
        )

        sed.add_info(
            "stellar.lum_old",
            lum_old,
            True,
            unit="W",
        )

        ### total population
        sed.add_info(
            "stellar.m_star",
            info_all["m_star"],
            True,
            unit="solMass",
        )

        sed.add_info(
            "stellar.m_gas",
            info_all["m_gas"],
            True,
            unit="solMass",
        )

        sed.add_info(
            "stellar.n_ly",
            info_all["n_ly"],
            True,
            unit="ph/s",
        )

        sed.add_info(
            "stellar.lum_ly",
            lum_lyc_young + lum_lyc_old,
            True,
            unit="W",
        )

        sed.add_info(
            "stellar.lum",
            lum_young + lum_old,
            True,
            unit="W",
        )

        sed.add_info(
            "stellar.age_m_star",
            info_all["age_mass"],
            unit="Myr",
        )

        sed.add_contribution(
            "stellar.old",
            wave,
            spec_old,
        )

        sed.add_contribution(
            "stellar.young",
            wave,
            spec_young,
        )


Module = SSPT