/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <mitkImageAccessByItk.h>

#include "mitkDescriptivePharmacokineticBrixModel.h"
#include "mitkDescriptivePharmacokineticBrixModelValueBasedParameterizer.h"

mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::StaticParameterMapType
mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::GetGlobalStaticParameters() const
{
  StaticParameterMapType result;
  StaticParameterValuesType values;
  values.push_back(m_Tau);
  result.insert(std::make_pair(ModelType::NAME_STATIC_PARAMETER_tau, values));
  return result;
};

mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::StaticParameterMapType
mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::GetLocalStaticParameters(
  const IndexType&) const
{
  StaticParameterMapType result;

  StaticParameterValuesType values;
  values.push_back(m_BaseValue);
  result.insert(std::make_pair(ModelType::NAME_STATIC_PARAMETER_s0, values));

  return result;
};

mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::ParametersType
mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::GetDefaultInitialParameterization()
const
{
  ParametersType initialParameters;
  initialParameters.SetSize(4);
  initialParameters[mitk::DescriptivePharmacokineticBrixModel::POSITION_PARAMETER_A] = 1.0;
  initialParameters[mitk::DescriptivePharmacokineticBrixModel::POSITION_PARAMETER_kep] = 4.0;
  initialParameters[mitk::DescriptivePharmacokineticBrixModel::POSITION_PARAMETER_kel] = 0.2;
  initialParameters[mitk::DescriptivePharmacokineticBrixModel::POSITION_PARAMETER_BAT] = 1.0;

  return initialParameters;
}

mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::DescriptivePharmacokineticBrixModelValueBasedParameterizer()
{
};

mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::~DescriptivePharmacokineticBrixModelValueBasedParameterizer()
{
};
