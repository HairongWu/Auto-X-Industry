![MITK Logo][logo]

The [Medical Imaging Interaction Toolkit][mitk] (MITK) is a free open-source software
system for development of interactive medical image processing software. MITK
combines the [Insight Toolkit][itk] (ITK) and the [Visualization Toolkit][vtk] (VTK) with an application framework.

The links below provide high-level and reference documentation targeting different
usage scenarios:

 - Get a [high-level overview][mitk-overview] about MITK with pointers to further
   documentation
 - End-users looking for help with MITK applications should read the
   [MITK User Manual][mitk-usermanual]
 - Developers contributing to or using MITK, please see the [MITK Developer Manual][mitk-devmanual]
   as well as the [MITK API Reference][mitk-apiref]

See the [MITK homepage][mitk] for details.

Supported platforms
-------------------

MITK is a cross-platform C++ toolkit and officially supports:

 - Windows
 - Linux
 - macOS

For details, please read the [Supported Platforms][platforms] page.

### Build status of develop branch

[![Windows][windows-build-status]][cdash]
[![Ubuntu 20.04][ubuntu-20.04-build-status]][cdash]
[![Ubuntu 22.04][ubuntu-22.04-build-status]][cdash]
[![macOS 10.15 Catalina][macos-10.15-build-status]][cdash]
[![macOS 11 Big Sur][macos-11-build-status]][cdash]

We highly recommend to use the stable **master** branch instead. It is updated 1-2 times per month accompanied by curated [changelogs][changelog] and [snapshot installers][snapshot-installers].

License
-------

Copyright (c) [German Cancer Research Center (DKFZ)][dkfz]. All rights reserved.

MITK is available as free open-source software under a [3-clause BSD license][license].

Download
--------

The MITK source code and binaries for the *MitkWorkbench* application are released regularly according to the [MITK release cycle][release-cycle]. See the [Download][download] page for a list of releases.

The official MITK source code is available in the [MITK Git repository][phab_repo]. The Git clone command is

    git clone https://phabricator.mitk.org/source/mitk.git MITK

Active development takes place in the MITK develop branch and its usage is advised for advanced users only.

How to contribute
-----------------

Contributions of all kind are happily accepted. However, to make the contribution process as smooth as possible, please read the [How to contribute to MITK][contribute] page if you plan to contribute to MITK.

Build instructions
------------------

MITK uses [CMake][cmake] to configure a build tree. The following is a crash course about cloning, configuring, and building MITK on a Linux/Unix system:

    git clone https://phabricator.mitk.org/source/mitk.git MITK
    mkdir MITK-build
    cd MITK-build
    cmake ../MITK
    make -j4

Read the comprehensive [build instructions][build] page for details.

Useful links
------------

 - [Homepage][mitk]
 - [Download][download]
 - [Mailing list][mailinglist]
 - [Issue tracker][bugs]

[logo]: https://github.com/MITK/MITK/raw/master/mitk.png
[mitk]: https://www.mitk.org
[itk]: https://itk.org
[vtk]: https://vtk.org
[mitk-overview]: https://docs.mitk.org/2024.06/
[mitk-usermanual]: https://docs.mitk.org/2024.06/UserManualPortal.html
[mitk-devmanual]: https://docs.mitk.org/2024.06/DeveloperManualPortal.html
[mitk-apiref]: https://docs.mitk.org/2024.06/usergroup0.html
[platforms]: https://docs.mitk.org/2024.06/SupportedPlatformsPage.html
[dkfz]: https://www.dkfz.de
[license]: https://github.com/MITK/MITK/blob/master/LICENSE
[release-cycle]: https://www.mitk.org/MitkReleaseCycle
[download]: https://www.mitk.org/Download
[phab_repo]: https://phabricator.mitk.org/source/mitk/
[contribute]: https://www.mitk.org/How_to_contribute
[cmake]: https://www.cmake.org
[build]: https://docs.mitk.org/2024.06/BuildInstructionsPage.html
[mailinglist]: https://www.mitk.org/Mailinglist
[bugs]: https://phabricator.mitk.org/maniphest/
[cdash]: https://cdash.mitk.org/index.php?project=MITK
[changelog]: https://phabricator.mitk.org/w/mitk/changelog/
[snapshot-installers]: https://www.mitk.org/download/ci/snapshots/
[windows-build-status]: https://ci.mitk.org/buildStatus/icon?job=MITK%2FContinuous%2FWindows&subject=Windows
[ubuntu-22.04-build-status]: https://ci.mitk.org/buildStatus/icon?job=MITK%2FContinuous%2FUbuntu+22.04&subject=Ubuntu+22.04
[ubuntu-20.04-build-status]: https://ci.mitk.org/buildStatus/icon?job=MITK%2FContinuous%2FUbuntu+20.04&subject=Ubuntu+20.04
[macOS-10.15-build-status]: https://ci.mitk.org/buildStatus/icon?job=MITK%2FContinuous%2FmacOS+Catalina&subject=macOS+10.15+Catalina
[macOS-11-build-status]: https://ci.mitk.org/buildStatus/icon?job=MITK%2FContinuous%2FmacOS+Big+Sur&subject=macOS+11+Big+Sur
