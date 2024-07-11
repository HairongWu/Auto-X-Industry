/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkVectorProperty.h"
#include <nlohmann/json.hpp>

namespace mitk
{
  template <typename DATATYPE>
  bool VectorProperty<DATATYPE>::IsEqual(const BaseProperty &property) const
  {
    return this->m_PropertyContent == static_cast<const Self &>(property).m_PropertyContent;
  }

  template <typename DATATYPE>
  bool VectorProperty<DATATYPE>::Assign(const BaseProperty &property)
  {
    this->m_PropertyContent = static_cast<const Self &>(property).m_PropertyContent;
    return true;
  }

  template <typename DATATYPE>
  itk::LightObject::Pointer VectorProperty<DATATYPE>::InternalClone() const
  {
    itk::LightObject::Pointer result(new Self(*this));
    return result;
  }

  template <typename DATATYPE>
  std::string VectorProperty<DATATYPE>::GetValueAsString() const
  {
    const size_t displayBlockLength = 3;
    size_t beginningElementsCount = displayBlockLength;
    size_t endElementsCount = displayBlockLength;

    if (m_PropertyContent.size() <= 2 * displayBlockLength)
    {
      beginningElementsCount = m_PropertyContent.size();
      endElementsCount = 0;
    }

    // return either a block of all items
    // if the total number of maximum 2*displayBlockLength
    //
    // or return the first and last "displayBlockLength"
    // number of items separated by "[...]";
    std::stringstream string_collector;
    for (size_t i = 0; i < beginningElementsCount; i++)
      string_collector << m_PropertyContent[i] << "\n";
    if (endElementsCount)
      string_collector << "[... " << m_PropertyContent.size() - 2 * displayBlockLength << " more]\n";
    for (size_t i = m_PropertyContent.size() - endElementsCount; i < m_PropertyContent.size(); ++i)
      string_collector << m_PropertyContent[i] << "\n";

    std::string return_value = string_collector.str();

    // remove last '\n'
    if (!return_value.empty())
      return_value.erase(return_value.size() - 1);

    return return_value;
  }

  template <typename DATATYPE>
  void VectorProperty<DATATYPE>::SetValue(const VectorType &newValue)
  {
    m_PropertyContent = newValue;
  }

  template <typename DATATYPE>
  const typename VectorProperty<DATATYPE>::VectorType &VectorProperty<DATATYPE>::GetValue() const
  {
    return m_PropertyContent;
  }

  template <typename DATATYPE>
  bool VectorProperty<DATATYPE>::ToJSON(nlohmann::json& j) const
  {
    j = this->GetValue();
    return true;
  }

  template <typename DATATYPE>
  bool VectorProperty<DATATYPE>::FromJSON(const nlohmann::json& j)
  {
    this->SetValue(j.get<VectorType>());
    return true;
  }

  // Explicit instantiation for defined types.
  MITK_DEFINE_VECTOR_PROPERTY(double)
  MITK_DEFINE_VECTOR_PROPERTY(int)
  MITK_DEFINE_VECTOR_PROPERTY(unsigned int)
  MITK_DEFINE_VECTOR_PROPERTY(std::string)

} // namespace mitk
