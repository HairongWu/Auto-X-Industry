/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkMaximumCurveDescriptionParameter_h
#define mitkMaximumCurveDescriptionParameter_h

#include "mitkCurveDescriptionParameterBase.h"

namespace mitk {

  /** Descriptor computes the maximum of the curve.*/
class MITKPHARMACOKINETICS_EXPORT MaximumCurveDescriptionParameter : public mitk::CurveDescriptionParameterBase
{
public:
    typedef mitk::MaximumCurveDescriptionParameter Self;
    typedef CurveDescriptionParameterBase Superclass;
    typedef itk::SmartPointer< Self >                            Pointer;
    typedef itk::SmartPointer< const Self >                      ConstPointer;

    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);
    DescriptionParameterNamesType GetDescriptionParameterName() const override;

protected:
    static const std::string PARAMETER_NAME;

    MaximumCurveDescriptionParameter();
    ~MaximumCurveDescriptionParameter() override;

    DescriptionParameterResultsType ComputeCurveDescriptionParameter(const CurveType& curve, const CurveGridType& grid) const override;
};



}
#endif
