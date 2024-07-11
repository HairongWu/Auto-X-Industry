/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkStatisticsToMaskRelationRule_h
#define mitkStatisticsToMaskRelationRule_h

#include <MitkImageStatisticsExports.h>
#include "mitkGenericIDRelationRule.h"

namespace mitk
{
  class MITKIMAGESTATISTICS_EXPORT StatisticsToMaskRelationRule : public mitk::GenericIDRelationRule
  {
  public:
    mitkClassMacro(StatisticsToMaskRelationRule, mitk::GenericIDRelationRule);
    itkNewMacro(Self);

  protected:
    StatisticsToMaskRelationRule();
  };
}

#endif
