/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <mitkLabelSetImageHelper.h>

#include <mitkLabelSetImage.h>
#include <mitkExceptionMacro.h>
#include <mitkProperties.h>

#include <array>
#include <regex>
#include <vector>

namespace
{
  template <typename T>
  std::array<int, 3> QuantizeColor(const T* color)
  {
    return {
      static_cast<int>(std::round(color[0] * 255)),
      static_cast<int>(std::round(color[1] * 255)),
      static_cast<int>(std::round(color[2] * 255)) };
  }

  mitk::Color FromLookupTableColor(const double* lookupTableColor)
  {
    mitk::Color color;
    color.Set(
      static_cast<float>(lookupTableColor[0]),
      static_cast<float>(lookupTableColor[1]),
      static_cast<float>(lookupTableColor[2]));
    return color;
  }
}

mitk::DataNode::Pointer mitk::LabelSetImageHelper::CreateEmptySegmentationNode(const std::string& segmentationName)
{
  auto newSegmentationNode = mitk::DataNode::New();
  newSegmentationNode->SetName(segmentationName);

  // initialize "showVolume"-property to false to prevent recalculating the volume while working on the segmentation
  newSegmentationNode->SetProperty("showVolume", mitk::BoolProperty::New(false));

  return newSegmentationNode;
}


mitk::DataNode::Pointer mitk::LabelSetImageHelper::CreateNewSegmentationNode(const DataNode* referenceNode,
  const Image* initialSegmentationImage, const std::string& segmentationName)
{
  std::string newSegmentationName = segmentationName;
  if (newSegmentationName.empty())
  {
    newSegmentationName = referenceNode->GetName();
    newSegmentationName.append("-labels");
  }

  if (nullptr == initialSegmentationImage)
  {
    return nullptr;
  }

  auto newLabelSetImage = mitk::LabelSetImage::New();
  try
  {
    newLabelSetImage->Initialize(initialSegmentationImage);
  }
  catch (mitk::Exception &e)
  {
    mitkReThrow(e) << "Could not initialize new label set image.";
    return nullptr;
  }

  auto newSegmentationNode = CreateEmptySegmentationNode(newSegmentationName);
  newSegmentationNode->SetData(newLabelSetImage);

  return newSegmentationNode;
}

mitk::Label::Pointer mitk::LabelSetImageHelper::CreateNewLabel(const LabelSetImage* labelSetImage, const std::string& namePrefix, bool hideIDIfUnique)
{
  if (nullptr == labelSetImage)
    return nullptr;

  const std::regex genericLabelNameRegEx(namePrefix + " ([1-9][0-9]*)");
  int maxGenericLabelNumber = 0;

  std::vector<std::array<int, 3>> colorsInUse = { {0,0,0} }; //black is always in use.

  for (auto & label : labelSetImage->GetLabels())
  {
    auto labelName = label->GetName();
    std::smatch match;

    if (std::regex_match(labelName, match, genericLabelNameRegEx))
      maxGenericLabelNumber = std::max(maxGenericLabelNumber, std::stoi(match[1].str()));

    const auto quantizedLabelColor = QuantizeColor(label->GetColor().data());

    if (std::find(colorsInUse.begin(), colorsInUse.end(), quantizedLabelColor) == std::end(colorsInUse))
      colorsInUse.push_back(quantizedLabelColor);
  }

  auto newLabel = mitk::Label::New();
  if (hideIDIfUnique && 0 == maxGenericLabelNumber)
  {
    newLabel->SetName(namePrefix);
  }
  else
  {
    newLabel->SetName(namePrefix + " " + std::to_string(maxGenericLabelNumber + 1));
  }

  auto lookupTable = mitk::LookupTable::New();
  lookupTable->SetType(mitk::LookupTable::LookupTableType::MULTILABEL);

  std::array<double, 3> lookupTableColor;
  const int maxTries = 25;
  bool newColorFound = false;

  for (int i = 0; i < maxTries; ++i)
  {
    lookupTable->GetColor(i, lookupTableColor.data());

    auto quantizedLookupTableColor = QuantizeColor(lookupTableColor.data());

    if (std::find(colorsInUse.begin(), colorsInUse.end(), quantizedLookupTableColor) == std::end(colorsInUse))
    {
      newLabel->SetColor(FromLookupTableColor(lookupTableColor.data()));
      newColorFound = true;
      break;
    }
  }

  if (!newColorFound)
  {
    lookupTable->GetColor(labelSetImage->GetTotalNumberOfLabels(), lookupTableColor.data());
    newLabel->SetColor(FromLookupTableColor(lookupTableColor.data()));
  }

  return newLabel;
}

mitk::LabelSetImageHelper::GroupIDToLabelValueMapType
mitk::LabelSetImageHelper::SplitLabelValuesByGroup(const LabelSetImage* labelSetImage, const LabelSetImage::LabelValueVectorType& labelValues)
{
  if (nullptr == labelSetImage)
    mitkThrow() << "Cannot split label values. Invalid LabelSetImage pointer passed";

  GroupIDToLabelValueMapType result;

  for (auto value : labelValues)
  {
    auto groupID = labelSetImage->GetGroupIndexOfLabel(value);

    //if groupID does not exist in result this call will also init an empty vector.
    result[groupID].push_back(value);
  }

  return result;
}

mitk::LabelSetImageHelper::LabelClassNameToLabelValueMapType
mitk::LabelSetImageHelper::SplitLabelValuesByClassNamwe(const LabelSetImage* labelSetImage, LabelSetImage::GroupIndexType groupID)
{
  if (nullptr == labelSetImage)
    mitkThrow() << "Cannot split label values. Invalid LabelSetImage pointer passed";

  return SplitLabelValuesByClassNamwe(labelSetImage, groupID, labelSetImage->GetLabelValuesByGroup(groupID));
}

mitk::LabelSetImageHelper::LabelClassNameToLabelValueMapType
mitk::LabelSetImageHelper::SplitLabelValuesByClassNamwe(const LabelSetImage* labelSetImage, LabelSetImage::GroupIndexType groupID, const LabelSetImage::LabelValueVectorType& labelValues)
{
  if (nullptr == labelSetImage)
    mitkThrow() << "Cannot split label values. Invalid LabelSetImage pointer passed";

  LabelClassNameToLabelValueMapType result;

  for (const auto value : labelValues)
  {
    if (labelSetImage->GetGroupIndexOfLabel(value) == groupID)
    {
      auto className = labelSetImage->GetLabel(value)->GetName();

      //if className does not exist in result this call will also init an empty vector.
      result[className].push_back(value);
    }
  }

  return result;
}

std::string mitk::LabelSetImageHelper::CreateDisplayGroupName(const LabelSetImage* labelSetImage, LabelSetImage::GroupIndexType groupID)
{
  const auto groupName = labelSetImage->GetGroupName(groupID);
  if (groupName.empty())
    return "Group "+std::to_string(groupID + 1);

  return groupName;
}
