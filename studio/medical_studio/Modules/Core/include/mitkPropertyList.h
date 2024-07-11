/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkPropertyList_h
#define mitkPropertyList_h

#include <mitkIPropertyOwner.h>
#include <mitkGenericProperty.h>

#include <nlohmann/json_fwd.hpp>

namespace mitk
{
  /**
   * @brief Key-value list holding instances of BaseProperty
   *
   * This list is meant to hold an arbitrary list of "properties",
   * which should describe the object associated with this list.
   *
   * Usually you will use PropertyList as part of a DataNode
   * object - in this context the properties describe the data object
   * held by the DataNode (e.g. whether the object is rendered at
   * all, which color is used for rendering, what name should be
   * displayed for the object, etc.)
   *
   * The values in the list are not fixed, you may introduce any kind
   * of property that seems useful - all you have to do is inherit
   * from BaseProperty.
   *
   * The list is organized as a key-value pairs, i.e.
   *
   *   \li "name" : pointer to a StringProperty
   *   \li "visible" : pointer to a BoolProperty
   *   \li "color" : pointer to a ColorProperty
   *   \li "volume" : pointer to a FloatProperty
   *
   * Please see the documentation of SetProperty and ReplaceProperty for two
   * quite different semantics. Normally SetProperty is what you want - this
   * method will try to change the value of an existing property and will
   * not allow you to replace e.g. a ColorProperty with an IntProperty.
   *
   * Please also regard, that the key of a property must be a none empty string.
   * This is a precondition. Setting properties with empty keys will raise an exception.
   *
   * @ingroup DataManagement
   */
  class MITKCORE_EXPORT PropertyList : public itk::Object, public IPropertyOwner
  {
  public:
    mitkClassMacroItkParent(PropertyList, itk::Object);

    /**
     * Method for creation through the object factory.
     */
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

    /**
     * Map structure to hold the properties: the map key is a string,
     * the value consists of the actual property object (BaseProperty).
     */
    typedef std::map<std::string, BaseProperty::Pointer> PropertyMap;
    typedef std::pair<std::string, BaseProperty::Pointer> PropertyMapElementType;

    // IPropertyProvider
    BaseProperty::ConstPointer GetConstProperty(const std::string &propertyKey, const std::string &contextName = "", bool fallBackOnDefaultContext = true) const override;
    std::vector<std::string> GetPropertyKeys(const std::string &contextName = "", bool includeDefaultContext = false) const override;
    std::vector<std::string> GetPropertyContextNames() const override;

    // IPropertyOwner
    BaseProperty * GetNonConstProperty(const std::string &propertyKey, const std::string &contextName = "", bool fallBackOnDefaultContext = true) override;
    void SetProperty(const std::string &propertyKey, BaseProperty *property, const std::string &contextName = "", bool fallBackOnDefaultContext = false) override;
    void RemoveProperty(const std::string &propertyKey, const std::string &contextName = "", bool fallBackOnDefaultContext = false) override;

    /**
     * @brief Get a property by its name.
     */
    mitk::BaseProperty *GetProperty(const std::string &propertyKey) const;

    /**
     * @brief Set a property object in the list/map by reference.
     *
     * The actual OBJECT holding the value of the property is replaced by this function.
     * This is useful if you want to change the type of the property, like from BoolProperty to StringProperty.
     * Another use is to share one and the same property object among several PropertyList/DataNode objects, which
     * makes them appear synchronized.
     */
    void ReplaceProperty(const std::string &propertyKey, BaseProperty *property);

    /**
     * @brief Set a property object in the list/map by reference.
     */
    void ConcatenatePropertyList(PropertyList *pList, bool replace = false);

    //##Documentation
    //## @brief Convenience access method for GenericProperty<T> properties
    //## (T being the type of the second parameter)
    //## @return @a true property was found
    template <typename T>
    bool GetPropertyValue(const char *propertyKey, T &value) const
    {
      GenericProperty<T> *gp = dynamic_cast<GenericProperty<T> *>(GetProperty(propertyKey));
      if (gp != nullptr)
      {
        value = gp->GetValue();
        return true;
      }
      return false;
    }

    /**
    * @brief Convenience method to access the value of a BoolProperty
    */
    bool GetBoolProperty(const char *propertyKey, bool &boolValue) const;
    /**
    * @brief ShortCut for the above method
    */
    bool Get(const char *propertyKey, bool &boolValue) const;

    /**
    * @brief Convenience method to set the value of a BoolProperty
    */
    void SetBoolProperty(const char *propertyKey, bool boolValue);
    /**
    * @brief ShortCut for the above method
    */
    void Set(const char *propertyKey, bool boolValue);

    /**
    * @brief Convenience method to access the value of an IntProperty
    */
    bool GetIntProperty(const char *propertyKey, int &intValue) const;
    /**
    * @brief ShortCut for the above method
    */
    bool Get(const char *propertyKey, int &intValue) const;

    /**
    * @brief Convenience method to set the value of an IntProperty
    */
    void SetIntProperty(const char *propertyKey, int intValue);
    /**
    * @brief ShortCut for the above method
    */
    void Set(const char *propertyKey, int intValue);

    /**
    * @brief Convenience method to access the value of a FloatProperty
    */
    bool GetFloatProperty(const char *propertyKey, float &floatValue) const;
    /**
    * @brief ShortCut for the above method
    */
    bool Get(const char *propertyKey, float &floatValue) const;

    /**
    * @brief Convenience method to set the value of a FloatProperty
    */
    void SetFloatProperty(const char *propertyKey, float floatValue);
    /**
    * @brief ShortCut for the above method
    */
    void Set(const char *propertyKey, float floatValue);

    /**
    * @brief Convenience method to access the value of a DoubleProperty
    */
    bool GetDoubleProperty(const char *propertyKey, double &doubleValue) const;
    /**
    * @brief ShortCut for the above method
    */
    bool Get(const char *propertyKey, double &doubleValue) const;

    /**
    * @brief Convenience method to set the value of a DoubleProperty
    */
    void SetDoubleProperty(const char *propertyKey, double doubleValue);
    /**
    * @brief ShortCut for the above method
    */
    void Set(const char *propertyKey, double doubleValue);

    /**
    * @brief Convenience method to access the value of a StringProperty
    */
    bool GetStringProperty(const char *propertyKey, std::string &stringValue) const;
    /**
    * @brief ShortCut for the above method
    */
    bool Get(const char *propertyKey, std::string &stringValue) const;

    /**
    * @brief Convenience method to set the value of a StringProperty
    */
    void SetStringProperty(const char *propertyKey, const char *stringValue);
    /**
    * @brief ShortCut for the above method
    */
    void Set(const char *propertyKey, const char *stringValue);
    /**
    * @brief ShortCut for the above method
    */
    void Set(const char *propertyKey, const std::string &stringValue);

    /**
     * @brief Get the timestamp of the last change of the map or the last change of one of
     * the properties store in the list (whichever is later).
     */
    itk::ModifiedTimeType GetMTime() const override;

    /**
     * @brief Remove a property from the list/map.
     */
    bool DeleteProperty(const std::string &propertyKey);

    const PropertyMap *GetMap() const { return &m_Properties; }
    bool IsEmpty() const { return m_Properties.empty(); }
    virtual void Clear();

    /**
     * @brief Serialize the property list to JSON.
     *
     * @note Properties of a certain type can only be deseralized again if their type has been
     * registered via the IPropertyDeserialization core service.
     *
     * @sa CoreServices
     * @sa IPropertyDeserialization::RegisterProperty
     */
    void ToJSON(nlohmann::json& j) const;

    /**
     * @brief Deserialize the property list from JSON.
     *
     * @note Properties of a certain type can only be deseralized again if their type has been
     * registered via the IPropertyDeserialization core service.
     *
     * @sa CoreServices
     * @sa IPropertyDeserialization::RegisterProperty
     */
    void FromJSON(const nlohmann::json& j);

  protected:
    PropertyList();
    PropertyList(const PropertyList &other);

    ~PropertyList() override;

    /**
     * @brief Map of properties.
     */
    PropertyMap m_Properties;

  private:
    itk::LightObject::Pointer InternalClone() const override;
  };

} // namespace mitk

#endif
