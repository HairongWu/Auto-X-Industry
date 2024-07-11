/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef mitkIsoDoseLevelVectorProperty_h
#define mitkIsoDoseLevelVectorProperty_h

#include "mitkBaseProperty.h"
#include "mitkIsoDoseLevelCollections.h"
#include "MitkRTExports.h"

namespace mitk {

/**
\brief Property class for dose iso level vector.
*/
class MITKRT_EXPORT IsoDoseLevelVectorProperty : public BaseProperty
{

protected:
    IsoDoseLevelVector::Pointer m_IsoLevelVector;

    IsoDoseLevelVectorProperty();

    explicit IsoDoseLevelVectorProperty(const IsoDoseLevelVectorProperty& other);

    explicit IsoDoseLevelVectorProperty(IsoDoseLevelVector* levelVector);

public:
    mitkClassMacro(IsoDoseLevelVectorProperty, BaseProperty);

    itkNewMacro(IsoDoseLevelVectorProperty);
    mitkNewMacro1Param(IsoDoseLevelVectorProperty, IsoDoseLevelVector*);

    typedef IsoDoseLevelVector ValueType;

    ~IsoDoseLevelVectorProperty() override;

    const IsoDoseLevelVector * GetIsoDoseLevelVector() const;
    const IsoDoseLevelVector * GetValue() const;
    IsoDoseLevelVector * GetIsoDoseLevelVector();
    IsoDoseLevelVector * GetValue();


    void SetIsoDoseLevelVector(IsoDoseLevelVector* levelVector);
    void SetValue(IsoDoseLevelVector* levelVector);

    std::string GetValueAsString() const override;

    bool ToJSON(nlohmann::json& j) const override;
    bool FromJSON(const nlohmann::json& j) override;

    using BaseProperty::operator=;

private:

    itk::LightObject::Pointer InternalClone() const override;

    bool IsEqual(const BaseProperty& property) const override;
    bool Assign(const BaseProperty& property) override;

};

} // namespace mitk



#endif
