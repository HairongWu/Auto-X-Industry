/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkSourceImageRelationRule_h
#define mitkSourceImageRelationRule_h

#include "mitkPropertyRelationRuleBase.h"
#include "mitkImage.h"

namespace mitk
{
  /**This rule class can be used for relations that reference an image as source for a destination entity.
  (e.g. an image that is used to generate the relation source).
  The ID-layer is supported like for GenericIDRelations.
  So it can be used for all ID based relations between PropertyProviders
  that also implement the interface identifiable.
  In addition the rule uses the data-layer to deduce/define relations.
  For this layer it uses properties compliant to DICOM. Thus (1) the information is stored in
  a DICOM Source Image Sequence item (0x0008,0x2112) and (2) the destination must have properties
  DICOM SOP Instance UIDs (0x0008, 0x0018) and DICOM SOP Class UID (0x0008, 0x0016). If the destination
  does not have this properties, no connection can be made on the data-layer.
  @remark Please note that PropertyRelationRules and DICOM use the term "source" differently. The DICOM
  source (image) equals the PropertyRelationRule destination. This is due to
  an inverted relation direction. So in the context of the SourceImageRelationRule interface a derived data is
  the source and points to the original image, it derives from. In the context of DICOM this referenced original image would be
  called source image (as the name of this class).
  In order to be able to use this class for different relation types (DICOM would call it purposes),
  the purposeTag is used. It must be specified when creating a rule instance.
  The purposeTag will be used as suffix for the rule ID of the instance and therefore allows to create
  specific and distinguishable rules instances based on this class.
  One may also specify the display name and the role names of the instance. If not specified the default values
  are used (display name: "<purposeTag> relation", source role name: "derived data",
  destination role name: "source image")
  */
  class MITKCORE_EXPORT SourceImageRelationRule : public mitk::PropertyRelationRuleBase
  {
  public:
    mitkClassMacro(SourceImageRelationRule, PropertyRelationRuleBase);
    itkNewMacro(Self);
    mitkNewMacro1Param(Self, const RuleIDType &);
    mitkNewMacro2Param(Self, const RuleIDType &, const std::string &);
    mitkNewMacro4Param(Self, const RuleIDType &, const std::string &, const std::string &, const std::string &);

    using RuleIDType = PropertyRelationRuleBase::RuleIDType;
    using RelationUIDType = PropertyRelationRuleBase::RelationUIDType;
    using RelationUIDVectorType = PropertyRelationRuleBase::RelationUIDVectorType;

    /** Returns an ID string that identifies the rule class */
    RuleIDType GetRuleID() const override;

    bool IsAbstract() const override;

    /** Returns a human readable string that can be used to describe the rule. Does not need to be unique.*/
    std::string GetDisplayName() const override;

    /** Returns a human readable string that can be used to describe the role of a source in context of the rule
     * instance.*/
    std::string GetSourceRoleName() const override;
    /** Returns a human readable string that can be used to describe the role of a destination in context of the rule
     * instance.*/
    std::string GetDestinationRoleName() const override;

    bool IsDestinationCandidate(const IPropertyProvider *owner) const override;

    /** Connects to passed images.
    @remark destination must specify DICOM SOP Instance UIDs (0x0008, 0x0018) and DICOM SOP Class UID (0x0008, 0x0016)
    in order to establish a connection on the data layer.*/
    RelationUIDType Connect(Image *source, const Image *destination) const;

  protected:
    SourceImageRelationRule();
    SourceImageRelationRule(const RuleIDType &purposeTag);
    SourceImageRelationRule(const RuleIDType &purposeTag, const std::string &displayName);
    SourceImageRelationRule(const RuleIDType &purposeTag,
                          const std::string &displayName,
                          const std::string &sourceRole,
                          const std::string &destinationRole);
    ~SourceImageRelationRule() override = default;

    using InstanceIDType = PropertyRelationRuleBase::InstanceIDType;
    using InstanceIDVectorType = PropertyRelationRuleBase::InstanceIDVectorType;

    /** Helper function that returns a vector of all selections of the property DICOM.0008.2112 (and its associated ruleID) that refer to destination or all (if no destination is passed) with a supported RuleID
     and which are not already covered by the ignoreInstances.*/
    std::vector<std::pair<size_t, std::string> > GetReferenceSequenceIndices(const IPropertyProvider * source,
      const IPropertyProvider* destination = nullptr, InstanceIDVectorType ignoreInstances = {}) const;

    using DataRelationUIDVectorType = PropertyRelationRuleBase::DataRelationUIDVectorType;
    virtual DataRelationUIDVectorType GetRelationUIDs_DataLayer(const IPropertyProvider* source,
      const IPropertyProvider* destination, const InstanceIDVectorType& instances_IDLayer) const override;

    void Connect_datalayer(IPropertyOwner *source,
                                   const IPropertyProvider *destination,
                                   const InstanceIDType &instanceID) const override;

    void Disconnect_datalayer(IPropertyOwner *source, const RelationUIDType& relationUID) const override;

    bool IsSupportedRuleID(const RuleIDType& ruleID) const override;

    itk::LightObject::Pointer InternalClone() const override;

    /**Prepares a new reference to an image on the data layer. Therefore an unused and valid sequence item index
    for the passed source will be generated and a relationUID property with the relationUID will be set to block the instance ID. The
    instance ID will be returned.
    @remark The method is guarded by a class wide mutex to avoid racing conditions in a scenario where rules are used
    concurrently.*/
    PropertyKeyPath::ItemSelectionIndex CreateNewSourceImageSequenceItem(IPropertyOwner *source) const;

    std::string GenerateRuleID(const std::string& purpose) const;

  private:
    RuleIDType m_PurposeTag;
    std::string m_DisplayName;
    std::string m_SourceRole;
    std::string m_DestinationRole;
  };

} // namespace mitk

#endif
