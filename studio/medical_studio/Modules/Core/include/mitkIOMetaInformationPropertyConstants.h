/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkIOMetaInformationPropertyConstants_h
#define mitkIOMetaInformationPropertyConstants_h

#include <MitkCoreExports.h>

#include "mitkPropertyKeyPath.h"

namespace mitk
{
  /**
   * @ingroup IO
   * @brief The IOMetaInformationPropertyConstants struct
   */
  struct MITKCORE_EXPORT IOMetaInformationPropertyConstants
  {
    //Path to the property containing the name of the reader used
    static PropertyKeyPath READER_DESCRIPTION();
    //Path to the property containing the version of mitk used to read the data
    static PropertyKeyPath READER_VERSION();
    //Path to the property containing the mine name detected used to read the data
    static PropertyKeyPath READER_MIME_NAME();
    //Path to the property containing the mime category detected to read the data
    static PropertyKeyPath READER_MIME_CATEGORY();
    //Path to the property containing the input location if loaded by file used to read the data
    static PropertyKeyPath READER_INPUTLOCATION();
    //Path to the properties containing the reader options used to read the data
    static PropertyKeyPath READER_OPTION_ROOT();
    static PropertyKeyPath READER_OPTIONS_ANY();
  };
}

#endif
