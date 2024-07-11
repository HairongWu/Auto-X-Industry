/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkOneTissueCompartmentModel.h"
#include "mitkConvolutionHelper.h"
#include <vnl/algo/vnl_fft_1d.h>
#include <fstream>

const std::string mitk::OneTissueCompartmentModel::MODEL_DISPLAY_NAME = "One Tissue Compartment Model";

const std::string mitk::OneTissueCompartmentModel::NAME_PARAMETER_K1 = "K_1";
const std::string mitk::OneTissueCompartmentModel::NAME_PARAMETER_k2 = "k_2";

const std::string mitk::OneTissueCompartmentModel::UNIT_PARAMETER_K1 = "1/min";
const std::string mitk::OneTissueCompartmentModel::UNIT_PARAMETER_k2 = "1/min";

const unsigned int mitk::OneTissueCompartmentModel::POSITION_PARAMETER_K1 = 0;
const unsigned int mitk::OneTissueCompartmentModel::POSITION_PARAMETER_k2 = 1;

const unsigned int mitk::OneTissueCompartmentModel::NUMBER_OF_PARAMETERS = 2;

const std::string mitk::OneTissueCompartmentModel::MODEL_TYPE = "Dynamic.PET";

std::string mitk::OneTissueCompartmentModel::GetModelDisplayName() const
{
  return MODEL_DISPLAY_NAME;
};

std::string mitk::OneTissueCompartmentModel::GetModelType() const
{
  return MODEL_TYPE;
};

mitk::OneTissueCompartmentModel::OneTissueCompartmentModel()
{

}

mitk::OneTissueCompartmentModel::~OneTissueCompartmentModel()
{

}

mitk::OneTissueCompartmentModel::ParameterNamesType mitk::OneTissueCompartmentModel::GetParameterNames() const
{
  ParameterNamesType result;

  result.push_back(NAME_PARAMETER_K1);
  result.push_back(NAME_PARAMETER_k2);

  return result;
}

mitk::OneTissueCompartmentModel::ParametersSizeType  mitk::OneTissueCompartmentModel::GetNumberOfParameters()
const
{
  return NUMBER_OF_PARAMETERS;
}


mitk::OneTissueCompartmentModel::ParamterUnitMapType
mitk::OneTissueCompartmentModel::GetParameterUnits() const
{
  ParamterUnitMapType result;

  result.insert(std::make_pair(NAME_PARAMETER_K1, UNIT_PARAMETER_K1));
  result.insert(std::make_pair(NAME_PARAMETER_k2, UNIT_PARAMETER_k2));

  return result;
};


mitk::OneTissueCompartmentModel::ModelResultType mitk::OneTissueCompartmentModel::ComputeModelfunction(
  const ParametersType& parameters) const
{
  if (this->m_TimeGrid.GetSize() == 0)
  {
    itkExceptionMacro("No Time Grid Set! Cannot Calculate Signal");
  }

  AterialInputFunctionType aterialInputFunction;
  aterialInputFunction = GetAterialInputFunction(this->m_TimeGrid);



  unsigned int timeSteps = this->m_TimeGrid.GetSize();

  //Model Parameters
  double     K1 = (double) parameters[POSITION_PARAMETER_K1] / 60.0;
  double     k2 = (double) parameters[POSITION_PARAMETER_k2] / 60.0;



  mitk::ModelBase::ModelResultType convolution = mitk::convoluteAIFWithExponential(this->m_TimeGrid,
      aterialInputFunction, k2);

  //Signal that will be returned by ComputeModelFunction
  mitk::ModelBase::ModelResultType signal(timeSteps);
  signal.fill(0.0);

  mitk::ModelBase::ModelResultType::iterator signalPos = signal.begin();



  for (mitk::ModelBase::ModelResultType::const_iterator res = convolution.begin(); res != convolution.end(); ++res, ++signalPos)
  {
    *signalPos = K1 * (*res);
  }

  return signal;

}




itk::LightObject::Pointer mitk::OneTissueCompartmentModel::InternalClone() const
{
  OneTissueCompartmentModel::Pointer newClone = OneTissueCompartmentModel::New();

  newClone->SetTimeGrid(this->m_TimeGrid);

  return newClone.GetPointer();
};

void mitk::OneTissueCompartmentModel::PrintSelf(std::ostream& os, ::itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);


};

