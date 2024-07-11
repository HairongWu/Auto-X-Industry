/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef _EXAMPLES_EXPORT_DLL_H_
#define _EXAMPLES_EXPORT_DLL_H_

//
// The following block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the org_mitk_gui_qt_examples_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// org_mitk_gui_qt_examples_EXPORTS functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
//
#if defined(_WIN32) && !defined(MITK_STATIC)
#if defined(org_mitk_gui_qt_examples_EXPORTS)
#define EXAMPLES_EXPORT __declspec(dllexport)
#else
#define EXAMPLES_EXPORT __declspec(dllimport)
#endif
#endif

#if !defined(EXAMPLES_EXPORT)
#define EXAMPLES_EXPORT
#endif

#endif /*_EXAMPLES_EXPORT_DLL_H_*/
