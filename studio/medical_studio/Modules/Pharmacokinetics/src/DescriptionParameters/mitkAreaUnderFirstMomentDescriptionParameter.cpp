/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkAreaUnderFirstMomentDescriptionParameter.h"

const std::string mitk::AreaUnderFirstMomentDescriptionParameter::PARAMETER_NAME = "AreaUnderFirstMoment";

mitk::AreaUnderFirstMomentDescriptionParameter::AreaUnderFirstMomentDescriptionParameter()
{
}

mitk::AreaUnderFirstMomentDescriptionParameter::~AreaUnderFirstMomentDescriptionParameter()
{
}

mitk::CurveDescriptionParameterBase::DescriptionParameterNamesType mitk::AreaUnderFirstMomentDescriptionParameter::GetDescriptionParameterName() const
{
  DescriptionParameterNamesType result = { PARAMETER_NAME };
  return result;
}

mitk::CurveDescriptionParameterBase::DescriptionParameterResultsType mitk::AreaUnderFirstMomentDescriptionParameter::ComputeCurveDescriptionParameter(const CurveType &curve, const CurveGridType &grid) const
{
  DescriptionParameterResultsType results;
  CurveDescriptionParameterResultType aumc = 0;
  auto steps = grid.GetSize();

  for (CurveGridType::size_type i = 0; i < steps - 1; ++i)
  {
    double deltaX = grid(i + 1) - grid(i);
    double deltaY = curve(i + 1)*grid(i + 1) - curve(i)*grid(i);
    double Yi = curve(i)*grid(i);

    double intI = 0.5 * deltaX * deltaY + Yi * deltaX;
    aumc += std::abs(intI);
  }
  results.push_back(aumc);
  return results;
}
