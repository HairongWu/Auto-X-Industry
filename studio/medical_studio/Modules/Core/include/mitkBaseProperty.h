/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkBaseProperty_h
#define mitkBaseProperty_h

#include <MitkCoreExports.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <string>
#include <nlohmann/json_fwd.hpp>

namespace mitk
{
  /*! \brief Abstract base class for properties

    \ingroup DataManagement

      Base class for properties. Properties are arbitrary additional information
      (to define a new type of information you have to define a subclass of
      BaseProperty) that can be added to a PropertyList.
      Concrete subclasses of BaseProperty should define Set-/Get-methods to assess
      the property value, which should be stored by value (not by reference).
      Subclasses must implement an operator==(const BaseProperty& property), which
      is used by PropertyList to check whether a property has been changed.
  */
  class MITKCORE_EXPORT BaseProperty : public itk::Object
  {
  public:
    mitkClassMacroItkParent(BaseProperty, itk::Object);
    itkCloneMacro(Self);

      /*! @brief Subclasses must implement IsEqual(const BaseProperty&) to support comparison.

          operator== which is used by PropertyList to check whether a property has been changed.
      */
      bool
      operator==(const BaseProperty &property) const;

    /*! @brief Assigns property to this BaseProperty instance.

        Subclasses must implement Assign(const BaseProperty&) and call the superclass
        Assign method for proper handling of polymorphic assignments. The assignment
        operator of the subclass should be disabled and the baseclass operator should
        be made visible using "using" statements.
    */
    BaseProperty &operator=(const BaseProperty &property);

    /*! @brief Assigns property to this BaseProperty instance.

        This method is identical to the assignment operator, except for the return type.
        It allows to directly check if the assignment was successful.
    */
    bool AssignProperty(const BaseProperty &property);

    virtual std::string GetValueAsString() const;

    /** \brief Serialize property value(s) to JSON.
     *
     * Rely on exceptions for error handling when implementing serialization.
     *
     * \return False if not serializable by design, true otherwise.
     */
    virtual bool ToJSON(nlohmann::json& j) const = 0;

    /** \brief Deserialize property value(s) from JSON.
    *
    * Rely on exceptions for error handling when implementing deserialization.
    *
    * \return False if not deserializable by design, true otherwise.
    */
    virtual bool FromJSON(const nlohmann::json& j) = 0;

    /**
     * @brief Default return value if a property which can not be returned as string
     */
    static const std::string VALUE_CANNOT_BE_CONVERTED_TO_STRING;

  protected:
    BaseProperty();
    BaseProperty(const BaseProperty &other);

    ~BaseProperty() override;

  private:
    /*!
      Override this method in subclasses to implement a meaningful comparison. The property
      argument is guaranteed to be castable to the type of the implementing subclass.
    */
    virtual bool IsEqual(const BaseProperty &property) const = 0;

    /*!
      Override this method in subclasses to implement a meaningful assignment. The property
      argument is guaranteed to be castable to the type of the implementing subclass.

      @warning This is not yet exception aware/safe and if this method returns false,
               this property's state might be undefined.

      @return True if the argument could be assigned to this property.
     */
    virtual bool Assign(const BaseProperty &) = 0;
  };

} // namespace mitk

#endif
