/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <iterator>
#include <set>
#include <type_traits>

#include "mitkTemporoSpatialStringProperty.h"

#include <nlohmann/json.hpp>

using CondensedTimeKeyType = std::pair<mitk::TimeStepType, mitk::TimeStepType>;
using CondensedTimePointsType = std::map<CondensedTimeKeyType, std::string>;

using CondensedSliceKeyType = std::pair<mitk::TemporoSpatialStringProperty::IndexValueType, mitk::TemporoSpatialStringProperty::IndexValueType>;
using CondensedSlicesType = std::map<CondensedSliceKeyType, CondensedTimePointsType>;

namespace
{
  /** Helper function that checks if between an ID and a successive ID is no gap.*/
  template<typename TValue>
  bool isGap(const TValue& value, const TValue& successor)
  {
    return value<successor || value > successor + 1;
  }


  template<typename TNewKey, typename TNewValue, typename TMasterKey, typename TMasterValue, typename TCondensedContainer>
  void CheckAndCondenseElement(const TNewKey& newKeyMinID, const TNewValue& newValue, TMasterKey& masterKey, TMasterValue& masterValue, TCondensedContainer& condensedContainer)
  {
    if (newValue != masterValue
      || isGap(newKeyMinID, masterKey.second))
    {
      condensedContainer[masterKey] = masterValue;
      masterValue = newValue;
      masterKey.first = newKeyMinID;
    }
    masterKey.second = newKeyMinID;
  }

  /** Helper function that tries to condense the values of time points for a slice as much as possible and returns all slices with condensed timepoint values.*/
  CondensedSlicesType CondenseTimePointValuesOfProperty(const mitk::TemporoSpatialStringProperty* tsProp)
  {
    CondensedSlicesType uncondensedSlices;

    auto zs = tsProp->GetAvailableSlices();
    for (const auto z : zs)
    {
      CondensedTimePointsType condensedTimePoints;
      auto timePointIDs = tsProp->GetAvailableTimeSteps(z);
      CondensedTimeKeyType condensedKey = { timePointIDs.front(),timePointIDs.front() };
      auto refValue = tsProp->GetValue(timePointIDs.front(), z);

      for (const auto timePointID : timePointIDs)
      {
        const auto& newVal = tsProp->GetValue(timePointID, z);
        CheckAndCondenseElement(timePointID, newVal, condensedKey, refValue, condensedTimePoints);
      }
      condensedTimePoints[condensedKey] = refValue;
      uncondensedSlices[{ z, z }] = condensedTimePoints;
    }
    return uncondensedSlices;
  }
}

mitk::TemporoSpatialStringProperty::TemporoSpatialStringProperty(const char *s)
{
  if (s)
  {
    SliceMapType slices{{0, s}};

    m_Values.insert(std::make_pair(0, slices));
  }
}

mitk::TemporoSpatialStringProperty::TemporoSpatialStringProperty(const std::string &s)
{
  SliceMapType slices{{0, s}};

  m_Values.insert(std::make_pair(0, slices));
}

mitk::TemporoSpatialStringProperty::TemporoSpatialStringProperty(const TemporoSpatialStringProperty &other)
  : BaseProperty(other), m_Values(other.m_Values)
{
}

bool mitk::TemporoSpatialStringProperty::IsEqual(const BaseProperty &property) const
{
  return this->m_Values == static_cast<const Self &>(property).m_Values;
}

bool mitk::TemporoSpatialStringProperty::Assign(const BaseProperty &property)
{
  this->m_Values = static_cast<const Self &>(property).m_Values;
  return true;
}

std::string mitk::TemporoSpatialStringProperty::GetValueAsString() const
{
  return GetValue();
}

bool mitk::TemporoSpatialStringProperty::IsUniform() const
{
  auto refValue = this->GetValue();

  for (const auto& timeStep : m_Values)
  {
    auto finding = std::find_if_not(timeStep.second.begin(), timeStep.second.end(), [&refValue](const mitk::TemporoSpatialStringProperty::SliceMapType::value_type& val) { return val.second == refValue; });
    if (finding != timeStep.second.end())
    {
      return false;
    }
  }

  return true;
}

itk::LightObject::Pointer mitk::TemporoSpatialStringProperty::InternalClone() const
{
  itk::LightObject::Pointer result(new Self(*this));
  result->UnRegister();
  return result;
}

mitk::TemporoSpatialStringProperty::ValueType mitk::TemporoSpatialStringProperty::GetValue() const
{
  std::string result = "";

  if (!m_Values.empty())
  {
    if (!m_Values.begin()->second.empty())
    {
      result = m_Values.begin()->second.begin()->second;
    }
  }
  return result;
};

std::pair<bool, mitk::TemporoSpatialStringProperty::ValueType> mitk::TemporoSpatialStringProperty::CheckValue(
  const TimeStepType &timeStep, const IndexValueType &zSlice, bool allowCloseTime, bool allowCloseSlice) const
{
  std::string value = "";
  bool found = false;

  auto timeIter = m_Values.find(timeStep);
  auto timeEnd = m_Values.end();
  if (timeIter == timeEnd && allowCloseTime)
  { // search for closest time step (earlier preverd)
    timeIter = m_Values.upper_bound(timeStep);
    if (timeIter != m_Values.begin())
    { // there is a key lower than time step
      timeIter = std::prev(timeIter);
    }
  }

  if (timeIter != timeEnd)
  {
    const SliceMapType &slices = timeIter->second;

    auto sliceIter = slices.find(zSlice);
    auto sliceEnd = slices.end();
    if (sliceIter == sliceEnd && allowCloseSlice)
    { // search for closest slice (earlier preverd)
      sliceIter = slices.upper_bound(zSlice);
      if (sliceIter != slices.begin())
      { // there is a key lower than slice
        sliceIter = std::prev(sliceIter);
      }
    }

    if (sliceIter != sliceEnd)
    {
      value = sliceIter->second;
      found = true;
    }
  }

  return std::make_pair(found, value);
};

mitk::TemporoSpatialStringProperty::ValueType mitk::TemporoSpatialStringProperty::GetValue(const TimeStepType &timeStep,
                                                                                           const IndexValueType &zSlice,
                                                                                           bool allowCloseTime,
                                                                                           bool allowCloseSlice) const
{
  return CheckValue(timeStep, zSlice, allowCloseTime, allowCloseSlice).second;
};

mitk::TemporoSpatialStringProperty::ValueType mitk::TemporoSpatialStringProperty::GetValueBySlice(
  const IndexValueType &zSlice, bool allowClose) const
{
  return GetValue(0, zSlice, true, allowClose);
};

mitk::TemporoSpatialStringProperty::ValueType mitk::TemporoSpatialStringProperty::GetValueByTimeStep(
  const TimeStepType &timeStep, bool allowClose) const
{
  return GetValue(timeStep, 0, allowClose, true);
};

bool mitk::TemporoSpatialStringProperty::HasValue() const
{
  return !m_Values.empty();
};

bool mitk::TemporoSpatialStringProperty::HasValue(const TimeStepType &timeStep,
                                                  const IndexValueType &zSlice,
                                                  bool allowCloseTime,
                                                  bool allowCloseSlice) const
{
  return CheckValue(timeStep, zSlice, allowCloseTime, allowCloseSlice).first;
};

bool mitk::TemporoSpatialStringProperty::HasValueBySlice(const IndexValueType &zSlice, bool allowClose) const
{
  return HasValue(0, zSlice, true, allowClose);
};

bool mitk::TemporoSpatialStringProperty::HasValueByTimeStep(const TimeStepType &timeStep, bool allowClose) const
{
  return HasValue(timeStep, 0, allowClose, true);
};

std::vector<mitk::TemporoSpatialStringProperty::IndexValueType> mitk::TemporoSpatialStringProperty::GetAvailableSlices() const
{
  std::set<IndexValueType> uniqueSlices;

  for (const auto& timeStep : m_Values)
  {
    for (const auto& slice : timeStep.second)
    {
      uniqueSlices.insert(slice.first);
    }
  }

  return std::vector<IndexValueType>(std::begin(uniqueSlices), std::end(uniqueSlices));
}

std::vector<mitk::TemporoSpatialStringProperty::IndexValueType> mitk::TemporoSpatialStringProperty::GetAvailableSlices(
  const TimeStepType &timeStep) const
{
  std::vector<IndexValueType> result;

  auto timeIter = m_Values.find(timeStep);
  auto timeEnd = m_Values.end();

  if (timeIter != timeEnd)
  {
    for (auto const &element : timeIter->second)
    {
      result.push_back(element.first);
    }
  }

  return result;
};

std::vector<mitk::TimeStepType> mitk::TemporoSpatialStringProperty::GetAvailableTimeSteps() const
{
  std::vector<mitk::TimeStepType> result;

  for (auto const &element : m_Values)
  {
    result.push_back(element.first);
  }

  return result;
};

std::vector<mitk::TimeStepType> mitk::TemporoSpatialStringProperty::GetAvailableTimeSteps(const IndexValueType& slice) const
{
  std::vector<mitk::TimeStepType> result;

  for (const auto& timeStep : m_Values)
  {
    if (timeStep.second.find(slice) != std::end(timeStep.second))
    {
      result.push_back(timeStep.first);
    }
  }
  return result;
}


void mitk::TemporoSpatialStringProperty::SetValue(const TimeStepType &timeStep,
                                                  const IndexValueType &zSlice,
                                                  const ValueType &value)
{
  auto timeIter = m_Values.find(timeStep);
  auto timeEnd = m_Values.end();

  if (timeIter == timeEnd)
  {
    SliceMapType slices{{zSlice, value}};
    m_Values.insert(std::make_pair(timeStep, slices));
  }
  else
  {
    timeIter->second[zSlice] = value;
  }
  this->Modified();
};

void mitk::TemporoSpatialStringProperty::SetValue(const ValueType &value)
{
  m_Values.clear();
  this->SetValue(0, 0, value);
};

bool mitk::TemporoSpatialStringProperty::ToJSON(nlohmann::json& j) const
{
  // We condense the content of the property to have a compact serialization.
  // We start with condensing time points and then slices (in difference to the
  // internal layout). Reason: There is more entropy in slices (looking at DICOM)
  // than across time points for one slice, so we can "compress" at a higher rate.
  // We didn't want to change the internal structure of the property as it would
  // introduce API inconvenience and subtle changes in behavior.
  auto uncondensedSlices = CondenseTimePointValuesOfProperty(this);
  CondensedSlicesType condensedSlices;

  if(!uncondensedSlices.empty())
  {
    auto& masterSlice = uncondensedSlices.begin()->second;
    auto masterSliceKey = uncondensedSlices.begin()->first;

    for (const auto& uncondensedSlice : uncondensedSlices)
    {
      const auto& uncondensedSliceID = uncondensedSlice.first.first;
      CheckAndCondenseElement(uncondensedSliceID, uncondensedSlice.second, masterSliceKey, masterSlice, condensedSlices);
    }

    condensedSlices[masterSliceKey] = masterSlice;
  }

  auto values = nlohmann::json::array();

  for (const auto& z : condensedSlices)
  {
    for (const auto& t : z.second)
    {
      const auto& minSliceID = z.first.first;
      const auto& maxSliceID = z.first.second;
      const auto& minTimePointID = t.first.first;
      const auto& maxTimePointID = t.first.second;

      auto value = nlohmann::json::object();
      value["z"] = minSliceID;

      if (minSliceID != maxSliceID)
        value["zmax"] = maxSliceID;

      value["t"] = minTimePointID;

      if (minTimePointID != maxTimePointID)
        value["tmax"] = maxTimePointID;

      value["value"] = t.second;

      values.push_back(value);
    }
  }

  j = nlohmann::json{{"values", values}};

  return true;
}

bool mitk::TemporoSpatialStringProperty::FromJSON(const nlohmann::json& j)
{
  for (const auto& element : j["values"])
  {
    auto value = element.value("value", "");
    auto z = element.value<TemporoSpatialStringProperty::IndexValueType>("z", 0);
    auto zmax = element.value<TemporoSpatialStringProperty::IndexValueType>("zmax", z);
    auto t = element.value<TimeStepType>("t", 0);
    auto tmax = element.value<TimeStepType>("tmax", t);

    for (auto currentT = t; currentT <= tmax; ++currentT)
    {
      for (auto currentZ = z; currentZ <= zmax; ++currentZ)
      {
        this->SetValue(currentT, currentZ, value);
      }
    }
  }

  return true;
}

std::string mitk::PropertyPersistenceSerialization::serializeTemporoSpatialStringPropertyToJSON(const mitk::BaseProperty *prop)
{
  const auto *tsProp = dynamic_cast<const mitk::TemporoSpatialStringProperty *>(prop);

  if (!tsProp)
    mitkThrow() << "Cannot serialize properties of types other than TemporoSpatialStringProperty.";

  nlohmann::json j;
  tsProp->ToJSON(j);

  return j.dump();
}

mitk::BaseProperty::Pointer mitk::PropertyPersistenceDeserialization::deserializeJSONToTemporoSpatialStringProperty(const std::string &value)
{
  if (value.empty())
    return nullptr;

  auto prop = mitk::TemporoSpatialStringProperty::New();

  auto root = nlohmann::json::parse(value);
  prop->FromJSON(root);

  return prop.GetPointer();
}
