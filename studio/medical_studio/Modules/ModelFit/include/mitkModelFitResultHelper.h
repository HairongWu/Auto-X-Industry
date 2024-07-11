/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkModelFitResultHelper_h
#define mitkModelFitResultHelper_h

#include <mitkImage.h>
#include <mitkDataNode.h>

#include "mitkScalarListLookupTableProperty.h"
#include "mitkModelBase.h"
#include "mitkModelFitParameter.h"
#include "mitkModelFitStaticParameterMap.h"

#include "MitkModelFitExports.h"

namespace mitk
{
  class DataStorage;
  class ModelTraitsInterface;
  class ModelBase;

  namespace modelFit
  {
    class ModelFitInfo;

    typedef std::map<ModelBase::ParameterNameType,Image::Pointer> ModelFitResultImageMapType;
    typedef std::vector<DataNode::Pointer> ModelFitResultNodeVectorType;

    /**Helper function that sets the properties of the passed base data according to the given model fit info instance and parameter specification.
     @param data Instance that properties should be configured.
     @param name Name of the parameter this data instance represents.
     @param dataType Type of the parameter this data instance represents.
     @param fitInfo Instance to the fit info that contains the information of the fit that derived the parameter.
     @pre Data must point to a valid instance.
     @pre fitInfo must point to an valid instance.
    */
    MITKMODELFIT_EXPORT void SetModelFitDataProperties(mitk::BaseData* data, const ModelBase::ParameterNameType& name, modelFit::Parameter::Type dataType, const modelFit::ModelFitInfo* fitInfo);

    MITKMODELFIT_EXPORT mitk::ScalarListLookupTableProperty::Pointer ConvertStaticParametersToProperty(const mitk::modelFit::StaticParameterMap& params);

    MITKMODELFIT_EXPORT DataNode::Pointer CreateResultNode(const ModelBase::ParameterNameType& name, modelFit::Parameter::Type nodeType, Image* parameterImage, const ModelFitInfo* modelFitInfo);

    MITKMODELFIT_EXPORT ModelFitResultNodeVectorType CreateResultNodeMap(const ModelFitResultImageMapType& results, const ModelFitResultImageMapType& derivedResults, const ModelFitResultImageMapType& criterionResults, const ModelFitResultImageMapType& evaluationResults, const ModelFitInfo* fitInfo);

    MITKMODELFIT_EXPORT void StoreResultsInDataStorage(DataStorage* storage, const ModelFitResultNodeVectorType& resultNodes, DataNode* parentNode = nullptr);

  }

}

#endif
