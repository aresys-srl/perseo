---
icon: lucide/cooking-pot
title: "Algorithm"
tags:
    - ambiguity ratio analysis
    - ptar
    - dtar
    - analysis
---

# Algorithm description

Target Ambiguity Ratio (TAR) analyses can be performed on Point Targets (**PTAR**) or Distributed Targets (**DTAR**) to
compute the ratio between the signal and its ambiguities, if captured by the SAR image.

## Point Target Ambiguity Ratio (PTAR)

This algorithms computes the ratio as:

$$
PTAR = 20\log_{10}\left(\frac{{|I_{amb_{left}}| + |I_{amb_{right}}|}}{2|I_{pt}|}\right)
$$

## Distributed Target Ambiguity Ratio (DTAR)

This algorithms computes the ratio as:

$$
DTAR = \frac{E(\Sigma |amb_{left}|^2) + E(\Sigma |amb_{right}|^2)}{2*E(\Sigma |target|^2)}
$$

## Graphical Output

Graphs can be generated from the analysis output using implemented features to obtain the target and ambiguities plots
and TAR value as show in the PTAR example below.

<figure markdown="span">
    ![PTAR](../../../../assets/images/q/ptar.png){ width="900" }
    <figcaption>Point Target Ambiguities and analysis results.</figcaption>
</figure>

!!! note "Graphical output"

    Graphical output functionalities are available only if the package has been installed with the ``[graphs]`` optional
    dependencies.  
    > :lucide-circle-chevron-right: Refer to the [installation documentation](../../../../install.md) for further information on how to install it.
