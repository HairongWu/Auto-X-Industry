/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkSmartPointerProperty_h
#define mitkSmartPointerProperty_h

#include "mitkBaseProperty.h"
#include "mitkUIDGenerator.h"
#include <MitkCoreExports.h>

#include <list>
#include <map>
#include <string>

namespace mitk
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif

  //##Documentation
  //## @brief Property containing a smart-pointer
  //## @ingroup DataManagement
  class MITKCORE_EXPORT SmartPointerProperty : public BaseProperty
  {
  public:
    mitkClassMacro(SmartPointerProperty, BaseProperty);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);
    mitkNewMacro1Param(SmartPointerProperty, itk::Object*);

    typedef itk::Object::Pointer ValueType;

    itk::Object::Pointer GetSmartPointer() const;
    ValueType GetValue() const;

    void SetSmartPointer(itk::Object *);
    void SetValue(const ValueType &);

    /// mainly for XML output
    std::string GetValueAsString() const override;

    static void PostProcessXMLReading();

    /// Return the number of SmartPointerProperties that reference the object given as parameter
    static unsigned int GetReferenceCountFor(itk::Object *);
    static std::string GetReferenceUIDFor(itk::Object *);
    static void RegisterPointerTarget(itk::Object *, const std::string uid);

    bool ToJSON(nlohmann::json& j) const override;
    bool FromJSON(const nlohmann::json& j) override;

    using BaseProperty::operator=;

  protected:
    SmartPointerProperty(itk::Object * = nullptr);
    SmartPointerProperty(const SmartPointerProperty &);

    itk::Object::Pointer m_SmartPointer;

  private:
    // purposely not implemented
    SmartPointerProperty &operator=(const SmartPointerProperty &);

    itk::LightObject::Pointer InternalClone() const override;

    bool IsEqual(const BaseProperty &) const override;
    bool Assign(const BaseProperty &) override;

    typedef std::map<itk::Object *, unsigned int> ReferenceCountMapType;
    typedef std::map<itk::Object *, std::string> ReferencesUIDMapType;
    typedef std::map<SmartPointerProperty *, std::string> ReadInSmartPointersMapType;
    typedef std::map<std::string, itk::Object *> ReadInTargetsMapType;

    /// for each itk::Object* count how many SmartPointerProperties point to it
    static ReferenceCountMapType m_ReferenceCount;
    static ReferencesUIDMapType m_ReferencesUID;
    static ReadInSmartPointersMapType m_ReadInInstances;
    static ReadInTargetsMapType m_ReadInTargets;

    /// to generate unique IDs for the objects pointed at (during XML writing)
    static UIDGenerator m_UIDGenerator;
  };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace mitk

#endif
