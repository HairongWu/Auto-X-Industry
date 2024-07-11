/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkGenericIDRelationRule.h"

bool mitk::GenericIDRelationRule::IsAbstract() const
{
  return m_RuleIDTag.empty();
};

bool mitk::GenericIDRelationRule::IsSupportedRuleID(const RuleIDType& ruleID) const
{
  return ruleID == this->GetRuleID() || (IsAbstract() && ruleID.find("IDRelation_") == 0);
};

mitk::GenericIDRelationRule::RuleIDType mitk::GenericIDRelationRule::GetRuleID() const
{
  return "IDRelation_" + m_RuleIDTag;
};

std::string mitk::GenericIDRelationRule::GetDisplayName() const
{
  return m_DisplayName;
};

std::string mitk::GenericIDRelationRule::GetSourceRoleName() const
{
  return m_SourceRole;
};

std::string mitk::GenericIDRelationRule::GetDestinationRoleName() const
{
  return m_DestinationRole;
};

mitk::GenericIDRelationRule::RelationUIDType mitk::GenericIDRelationRule::Connect(IPropertyOwner *source, const IPropertyProvider *destination) const
{
  return Superclass::Connect(source, destination);
};

mitk::GenericIDRelationRule::GenericIDRelationRule(const RuleIDType &ruleIDTag)
  : GenericIDRelationRule(ruleIDTag, ruleIDTag + " relation"){};

mitk::GenericIDRelationRule::GenericIDRelationRule(const RuleIDType &ruleIDTag, const std::string &displayName)
  : GenericIDRelationRule(
      ruleIDTag, displayName, "source of " + ruleIDTag + " relation", "destination of " + ruleIDTag + " relation"){};

mitk::GenericIDRelationRule::GenericIDRelationRule(const RuleIDType &ruleIDTag,
                                                   const std::string &displayName,
                                                   const std::string &sourceRole,
                                                   const std::string &destinationRole)
  : m_RuleIDTag(ruleIDTag), m_DisplayName(displayName), m_SourceRole(sourceRole), m_DestinationRole(destinationRole){};

mitk::GenericIDRelationRule::DataRelationUIDVectorType mitk::GenericIDRelationRule::GetRelationUIDs_DataLayer(
  const IPropertyProvider * /*source*/, const IPropertyProvider * /*destination*/, const InstanceIDVectorType& /*instances_IDLayer*/) const
{
  // Data layer is not supported by the class. Therefore return empty vector.
  return DataRelationUIDVectorType();
};

void mitk::GenericIDRelationRule::Connect_datalayer(IPropertyOwner * /*source*/,
                                                    const IPropertyProvider * /*destination*/,
                                                    const InstanceIDType & /*instanceID*/) const {
  // Data layer is not supported by the class. => Do nothing
};

void mitk::GenericIDRelationRule::Disconnect_datalayer(IPropertyOwner * /*source*/, const RelationUIDType & /*relationUID*/) const {
  // Data layer is not supported by the class. => Do nothing
};

itk::LightObject::Pointer mitk::GenericIDRelationRule::InternalClone() const
{
  itk::LightObject::Pointer result = Self::New(this->m_RuleIDTag, this->m_DisplayName, this->m_SourceRole, this->m_DestinationRole).GetPointer();

  return result;
};
