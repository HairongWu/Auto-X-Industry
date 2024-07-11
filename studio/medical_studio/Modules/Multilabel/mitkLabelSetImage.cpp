/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkLabelSetImage.h"

#include <mitkImageAccessByItk.h>
#include <mitkImageCast.h>
#include <mitkImagePixelWriteAccessor.h>
#include <mitkPadImageFilter.h>
#include <mitkDICOMSegmentationPropertyHelper.h>
#include <mitkDICOMQIPropertyHelper.h>
#include <mitkNodePredicateGeometry.h>

#include <itkLabelGeometryImageFilter.h>
#include <itkCommand.h>
#include <itkBinaryFunctorImageFilter.h>


namespace mitk
{
  template <typename ImageType>
  void ClearBufferProcessing(ImageType* itkImage)
  {
    itkImage->FillBuffer(0);
  }

  void ClearImageBuffer(mitk::Image* image)
  {
    if (image->GetDimension() == 4)
    { //remark: this extra branch was added, because LabelSetImage instances can be
      //dynamic (4D), but AccessByItk by support only supports 2D and 3D.
      //The option to change the CMake default dimensions for AccessByItk was
      //dropped (for details see discussion in T28756)
      AccessFixedDimensionByItk(image, ClearBufferProcessing, 4);
    }
    else
    {
      AccessByItk(image, ClearBufferProcessing);
    }
  }
}

const mitk::LabelSetImage::LabelValueType mitk::LabelSetImage::UNLABELED_VALUE = 0;

mitk::LabelSetImage::LabelSetImage()
  : mitk::Image(), m_ActiveLabelValue(0), m_UnlabeledLabelLock(false), m_ActiveLayer(0), m_activeLayerInvalid(false)
{
  m_LookupTable = mitk::LookupTable::New();
  m_LookupTable->SetType(mitk::LookupTable::MULTILABEL);

  // Add some DICOM Tags as properties to segmentation image
  DICOMSegmentationPropertyHelper::DeriveDICOMSegmentationProperties(this);
}

mitk::LabelSetImage::LabelSetImage(const mitk::LabelSetImage &other)
  : Image(other),
    m_ActiveLabelValue(other.m_ActiveLabelValue),
    m_LookupTable(other.m_LookupTable->Clone()),
    m_UnlabeledLabelLock(other.m_UnlabeledLabelLock),
    m_ActiveLayer(other.GetActiveLayer()),
    m_activeLayerInvalid(false)
{
  GroupIndexType i = 0;
  for (auto groupImage : other.m_LayerContainer)
  {
    this->AddLayer(groupImage->Clone(), other.GetConstLabelsByValue(other.GetLabelValuesByGroup(i)));
    i++;
  }
  m_Groups = other.m_Groups;

  // Add some DICOM Tags as properties to segmentation image
  DICOMSegmentationPropertyHelper::DeriveDICOMSegmentationProperties(this);
}

void mitk::LabelSetImage::Initialize(const mitk::Image *other)
{
  mitk::PixelType pixelType(mitk::MakeScalarPixelType<LabelSetImage::PixelType>());
  if (other->GetDimension() == 2)
  {
    const unsigned int dimensions[] = {other->GetDimension(0), other->GetDimension(1), 1};
    Superclass::Initialize(pixelType, 3, dimensions);
  }
  else
  {
    Superclass::Initialize(pixelType, other->GetDimension(), other->GetDimensions());
  }

  auto originalGeometry = other->GetTimeGeometry()->Clone();
  this->SetTimeGeometry(originalGeometry);

  // initialize image memory to zero
  ClearImageBuffer(this);

  // Transfer some general DICOM properties from the source image to derived image (e.g. Patient information,...)
  DICOMQIPropertyHelper::DeriveDICOMSourceProperties(other, this);

  // Add an initial LabelSet and corresponding image data to the stack
  if (this->GetNumberOfLayers() == 0)
  {
    AddLayer();
  }
}

mitk::LabelSetImage::~LabelSetImage()
{
  for (auto [value, label] : m_LabelMap)
  {
    (void)value; // Prevent unused variable error in older compilers
    this->ReleaseLabel(label);
  }
  m_LabelMap.clear();
}

unsigned int mitk::LabelSetImage::GetActiveLayer() const
{
  if (m_LayerContainer.size() == 0) mitkThrow() << "Cannot return active layer index. No layer is available.";

  return m_ActiveLayer;
}

unsigned int mitk::LabelSetImage::GetNumberOfLayers() const
{
  return m_LayerContainer.size();
}

void mitk::LabelSetImage::RemoveGroup(GroupIndexType indexToDelete)
{
  if (!this->ExistGroup(indexToDelete)) mitkThrow() << "Cannot remove group. Group does not exist. Invalid group index: "<<indexToDelete;

  const auto activeIndex = GetActiveLayer();

  auto newActiveIndex = activeIndex;
  auto newActiveIndexBeforeDeletion = activeIndex;
  //determine new active group index (after the group will be removed);
  if (indexToDelete < activeIndex)
  { //lower the index because position in m_LayerContainer etc has changed
    newActiveIndex = activeIndex-1;
  }
  else if (indexToDelete == activeIndex)
  {
    if (this->GetNumberOfLayers() == 1)
    { //last layer is about to be deleted
      newActiveIndex = 0;
    }
    else
    {
      //we have to add/subtract one more because we have not removed the layer yet, thus the group count is to 1 high.
      newActiveIndex = indexToDelete+1 < GetNumberOfLayers() ? indexToDelete : GetNumberOfLayers() - 2;
      newActiveIndexBeforeDeletion = indexToDelete + 1 < GetNumberOfLayers() ? indexToDelete+1 : indexToDelete -1;
    }
  }

  if (activeIndex == indexToDelete)
  {
    // we are deleting the active layer, it should not be copied back into the vector
    m_activeLayerInvalid = true;
    //copy the image content of the upcoming new active layer; 
    SetActiveLayer(newActiveIndexBeforeDeletion);
  }

  auto relevantLabels = m_GroupToLabelMap[indexToDelete];

  {
    std::lock_guard<std::shared_mutex> guard(m_LabelNGroupMapsMutex);
    // remove labels of group
    for (auto labelValue : relevantLabels)
    {
      auto label = m_LabelMap[labelValue];
      this->ReleaseLabel(label);
      m_LabelToGroupMap.erase(labelValue);
      m_LabelMap.erase(labelValue);
      this->InvokeEvent(LabelRemovedEvent(labelValue));
    }
    // remove the group entries in the maps and the image.
    m_Groups.erase(m_Groups.begin() + indexToDelete);
    m_GroupToLabelMap.erase(m_GroupToLabelMap.begin() + indexToDelete);
    m_LayerContainer.erase(m_LayerContainer.begin() + indexToDelete);
  }

  //update old indexes in m_GroupToLabelMap to new layer indexes
  for (auto& element : m_LabelToGroupMap)
  {
    if (element.second > indexToDelete) element.second = element.second -1;
  }

  //correct active layer index
  m_ActiveLayer = newActiveIndex;

  this->InvokeEvent(LabelsChangedEvent(relevantLabels));
  this->InvokeEvent(GroupRemovedEvent(indexToDelete));
  this->Modified();
}

mitk::LabelSetImage::LabelValueVectorType mitk::LabelSetImage::ExtractLabelValuesFromLabelVector(const LabelVectorType& labels)
{
  LabelValueVectorType result;

  for (auto label : labels)
  {
    result.emplace_back(label->GetValue());
  }
  return result;
}

mitk::LabelSetImage::LabelValueVectorType mitk::LabelSetImage::ExtractLabelValuesFromLabelVector(const ConstLabelVectorType& labels)
{
  LabelValueVectorType result;

  for (auto label : labels)
  {
    result.emplace_back(label->GetValue());
  }
  return result;
}

mitk::LabelSetImage::ConstLabelVectorType mitk::LabelSetImage::ConvertLabelVectorConst(const LabelVectorType& labels)
{
  ConstLabelVectorType result(labels.begin(), labels.end());
  return result;
};

const mitk::LabelSetImage::LabelValueVectorType mitk::LabelSetImage::GetAllLabelValues() const
{
  LabelValueVectorType result;

  for (auto [value, label] : m_LabelMap)
  {
    (void)label; // Prevent unused variable error in older compilers
    result.emplace_back(value);
  }
  return result;
}

mitk::LabelSetImage::LabelValueVectorType mitk::LabelSetImage::GetUsedLabelValues() const
{
  LabelValueVectorType result = { UNLABELED_VALUE };

  for (auto [value, label] : m_LabelMap)
  {
    (void)label; // Prevent unused variable error in older compilers
    result.emplace_back(value);
  }

  return result;
}

mitk::LabelSetImage::GroupIndexType mitk::LabelSetImage::AddLayer(ConstLabelVector labels)
{
  mitk::Image::Pointer newImage = mitk::Image::New();
  newImage->Initialize(this->GetPixelType(),
                       this->GetDimension(),
                       this->GetDimensions(),
                       this->GetImageDescriptor()->GetNumberOfChannels());
  newImage->SetTimeGeometry(this->GetTimeGeometry()->Clone());

  ClearImageBuffer(newImage);

  return this->AddLayer(newImage, labels);
}

mitk::LabelSetImage::GroupIndexType mitk::LabelSetImage::AddLayer(mitk::Image* layerImage, ConstLabelVector labels)
{
  GroupIndexType newGroupID = m_Groups.size();

  if (nullptr == layerImage)
    mitkThrow() << "Cannot add group. Passed group image is nullptr.";

  bool equalGeometries = Equal(
    *(this->GetTimeGeometry()),
    *(layerImage->GetTimeGeometry()),
    NODE_PREDICATE_GEOMETRY_DEFAULT_CHECK_COORDINATE_PRECISION,
    NODE_PREDICATE_GEOMETRY_DEFAULT_CHECK_DIRECTION_PRECISION,
    false);

  if (!equalGeometries)
    mitkThrow() << "Cannot add group. Passed group image has not the same geometry like segmentation.";

  if (layerImage->GetPixelType() != MakePixelType<LabelValueType, LabelValueType, 1>())
    mitkThrow() << "Cannot add group. Passed group image has incorrect pixel type. Only LabelValueType is supported. Invalid pixel type: "<< layerImage->GetPixelType().GetTypeAsString();

  // push a new working image for the new layer
  m_LayerContainer.push_back(layerImage);

  m_Groups.push_back("");
  m_GroupToLabelMap.push_back({});

  for (auto label : labels)
  {
    if (m_LabelMap.end() != m_LabelMap.find(label->GetValue()))
    {
      mitkThrow() << "Cannot add layer. Labels that should be added with layer use at least one label value that is already in use. Conflicted label value: " << label->GetValue();
    }

    auto labelClone = label->Clone();

    DICOMSegmentationPropertyHelper::SetDICOMSegmentProperties(labelClone);
    this->AddLabelToMap(labelClone->GetValue(), labelClone, newGroupID);
    this->RegisterLabel(labelClone);
  }

  this->Modified();
  this->InvokeEvent(GroupAddedEvent(newGroupID));

  return newGroupID;
}

void mitk::LabelSetImage::ReplaceGroupLabels(const GroupIndexType groupID, const ConstLabelVectorType& labelSet)
{
  if (m_LayerContainer.size() <= groupID)
  {
    mitkThrow() << "Trying to replace labels of non-existing group. Invalid group id: "<<groupID;
  }

  //remove old group labels
  LabelValueVectorType oldLabels;
  {
    std::lock_guard<std::shared_mutex> guard(m_LabelNGroupMapsMutex);
    oldLabels = this->m_GroupToLabelMap[groupID];
    for (auto labelID : oldLabels)
    {
      this->RemoveLabelFromMap(labelID);
      this->InvokeEvent(LabelRemovedEvent(labelID));

    }
  }
  this->InvokeEvent(LabelsChangedEvent(oldLabels));
  this->InvokeEvent(GroupModifiedEvent(groupID));

  //add new labels to group
  for (auto label : labelSet)
  {
    this->AddLabel(label->Clone(), groupID, true, false);
  }
}

void mitk::LabelSetImage::ReplaceGroupLabels(const GroupIndexType groupID, const LabelVectorType& labelSet)
{
  return ReplaceGroupLabels(groupID, ConvertLabelVectorConst(labelSet));
}

mitk::Image* mitk::LabelSetImage::GetGroupImage(GroupIndexType groupID)
{
  if (!this->ExistGroup(groupID)) mitkThrow() << "Error, cannot return group image. Group ID is invalid. Invalid ID: " << groupID;

  return groupID == this->GetActiveLayer() ? this : m_LayerContainer[groupID];
}


const mitk::Image* mitk::LabelSetImage::GetGroupImage(GroupIndexType groupID) const
{
  if (!this->ExistGroup(groupID)) mitkThrow() << "Error, cannot return group image. Group ID is invalid. Invalid ID: " << groupID;

  return groupID == this->GetActiveLayer() ? this : m_LayerContainer.at(groupID).GetPointer();
}

const mitk::Image* mitk::LabelSetImage::GetGroupImageWorkaround(GroupIndexType groupID) const
{
  if (!this->ExistGroup(groupID))
    mitkThrow() << "Error, cannot return group image. Group ID is invalid. Invalid ID: " << groupID;

  if (groupID == this->GetActiveLayer() && this->GetMTime()> m_LayerContainer[groupID]->GetMTime())
  { //we have to transfer the content first into the group image
    if (4 == this->GetDimension())
    {
      AccessFixedDimensionByItk_n(this, ImageToLayerContainerProcessing, 4, (groupID));
    }
    else
    {
      AccessByItk_1(this, ImageToLayerContainerProcessing, groupID);
    }
  }

  return m_LayerContainer[groupID].GetPointer();
}

const std::string& mitk::LabelSetImage::GetGroupName(GroupIndexType groupID) const
{
  if (!this->ExistGroup(groupID))
    mitkThrow() << "Error, cannot return group name. Group ID is invalid. Invalid ID: " << groupID;

  return m_Groups[groupID];
}

void mitk::LabelSetImage::SetGroupName(GroupIndexType groupID, const std::string& name)
{
  if (!this->ExistGroup(groupID))
    mitkThrow() << "Error, cannot set group name. Group ID is invalid. Invalid ID: " << groupID;

  m_Groups[groupID] = name;
  this->InvokeEvent(GroupModifiedEvent(groupID));
}

void mitk::LabelSetImage::SetActiveLayer(unsigned int layer)
{
  try
  {
    if (4 == this->GetDimension())
    {
      if ((layer != GetActiveLayer() || m_activeLayerInvalid) && (layer < this->GetNumberOfLayers()))
      {
        BeforeChangeLayerEvent.Send();

        if (m_activeLayerInvalid)
        {
          // We should not write the invalid layer back to the vector
          m_activeLayerInvalid = false;
        }
        else
        {
          AccessFixedDimensionByItk_n(this, ImageToLayerContainerProcessing, 4, (GetActiveLayer()));
        }
        m_ActiveLayer = layer;
        AccessFixedDimensionByItk_n(this, LayerContainerToImageProcessing, 4, (GetActiveLayer()));

        AfterChangeLayerEvent.Send();
      }
    }
    else
    {
      if ((layer != GetActiveLayer() || m_activeLayerInvalid) && (layer < this->GetNumberOfLayers()))
      {
        BeforeChangeLayerEvent.Send();

        if (m_activeLayerInvalid)
        {
          // We should not write the invalid layer back to the vector
          m_activeLayerInvalid = false;
        }
        else
        {
          AccessByItk_1(this, ImageToLayerContainerProcessing, GetActiveLayer());
        }
        m_ActiveLayer = layer;
        AccessByItk_1(this, LayerContainerToImageProcessing, GetActiveLayer());

        AfterChangeLayerEvent.Send();
      }
    }
  }
  catch (itk::ExceptionObject &e)
  {
    mitkThrow() << e.GetDescription();
  }
  this->Modified();
}

void mitk::LabelSetImage::SetActiveLabel(LabelValueType label)
{
  m_ActiveLabelValue = label;

  if (label != UNLABELED_VALUE)
  {
    auto groupID = this->GetGroupIndexOfLabel(label);
    if (groupID!=this->GetActiveLayer()) this->SetActiveLayer(groupID);
  }
  Modified();
}

void mitk::LabelSetImage::ClearBuffer()
{
  try
  {
    ClearImageBuffer(this);
    this->Modified();
  }
  catch (itk::ExceptionObject &e)
  {
    mitkThrow() << e.GetDescription();
  }
}

void mitk::LabelSetImage::MergeLabel(PixelType pixelValue, PixelType sourcePixelValue)
{
  try
  {
    AccessByItk_2(this, MergeLabelProcessing, pixelValue, sourcePixelValue);
  }
  catch (itk::ExceptionObject &e)
  {
    mitkThrow() << e.GetDescription();
  }
  this->SetActiveLabel(pixelValue);
  this->InvokeEvent(LabelModifiedEvent(sourcePixelValue));
  this->InvokeEvent(LabelModifiedEvent(pixelValue));
  this->InvokeEvent(LabelsChangedEvent({ sourcePixelValue, pixelValue }));
  Modified();
}

void mitk::LabelSetImage::MergeLabels(PixelType pixelValue, const std::vector<PixelType>& vectorOfSourcePixelValues)
{
  try
  {
    for (unsigned int idx = 0; idx < vectorOfSourcePixelValues.size(); idx++)
    {
      AccessByItk_2(this, MergeLabelProcessing, pixelValue, vectorOfSourcePixelValues[idx]);
      this->InvokeEvent(LabelModifiedEvent(vectorOfSourcePixelValues[idx]));
    }
  }
  catch (itk::ExceptionObject &e)
  {
    mitkThrow() << e.GetDescription();
  }
  this->SetActiveLabel(pixelValue);
  this->InvokeEvent(LabelModifiedEvent(pixelValue));
  auto modifiedValues = vectorOfSourcePixelValues;
  modifiedValues.push_back(pixelValue);
  this->InvokeEvent(LabelsChangedEvent(modifiedValues));

  Modified();
}

void mitk::LabelSetImage::RemoveLabel(LabelValueType pixelValue)
{
  GroupIndexType groupID = 0;
  {
    std::lock_guard<std::shared_mutex> guard(m_LabelNGroupMapsMutex);
    if (m_LabelMap.find(pixelValue) == m_LabelMap.end()) return;

    groupID = this->GetGroupIndexOfLabel(pixelValue);

    //first erase the pixel content (also triggers a LabelModified event)
    this->EraseLabel(pixelValue);
    this->RemoveLabelFromMap(pixelValue);


    if (m_ActiveLabelValue == pixelValue)
    {
      this->SetActiveLabel(0);
    }
  }

  this->InvokeEvent(LabelRemovedEvent(pixelValue));
  this->InvokeEvent(LabelsChangedEvent({ pixelValue }));
  this->InvokeEvent(GroupModifiedEvent(groupID));
}

void mitk::LabelSetImage::RemoveLabelFromMap(LabelValueType pixelValue)
{
  if (m_LabelMap.find(pixelValue) == m_LabelMap.end()) mitkThrow()<<"Invalid state of instance. RemoveLabelFromMap was called for unknown label id. invalid label id: "<<pixelValue;

  auto groupID = this->GetGroupIndexOfLabel(pixelValue);

  this->ReleaseLabel(m_LabelMap[pixelValue]);
  //now remove the label entry itself
  m_LabelMap.erase(pixelValue);
  m_LabelToGroupMap.erase(pixelValue);
  auto labelsInGroup = m_GroupToLabelMap[groupID];
  labelsInGroup.erase(std::remove(labelsInGroup.begin(), labelsInGroup.end(), pixelValue), labelsInGroup.end());
  m_GroupToLabelMap[groupID] = labelsInGroup;
}

void mitk::LabelSetImage::RemoveLabels(const LabelValueVectorType& vectorOfLabelPixelValues)
{
  for (const auto labelValue : vectorOfLabelPixelValues)
  {
    this->RemoveLabel(labelValue);
  }
  this->InvokeEvent(LabelsChangedEvent(vectorOfLabelPixelValues));
}

void mitk::LabelSetImage::EraseLabel(LabelValueType pixelValue)
{
  try
  {
    auto groupID = this->GetGroupIndexOfLabel(pixelValue);

    mitk::Image* groupImage = this->GetGroupImage(groupID);

    if (4 == this->GetDimension())
    {
      AccessFixedDimensionByItk_1(groupImage, EraseLabelProcessing, 4, pixelValue);
    }
    else
    {
      AccessByItk_1(groupImage, EraseLabelProcessing, pixelValue);
    }
    groupImage->Modified();
  }
  catch (const itk::ExceptionObject& e)
  {
    mitkThrow() << e.GetDescription();
  }

  this->InvokeEvent(LabelModifiedEvent(pixelValue));
  this->InvokeEvent(LabelsChangedEvent({ pixelValue }));
  Modified();
}

void mitk::LabelSetImage::EraseLabels(const LabelValueVectorType& labelValues)
{
  for (auto labelValue : labelValues)
  {
    this->EraseLabel(labelValue);
  }
}

mitk::LabelSetImage::LabelValueType mitk::LabelSetImage::GetUnusedLabelValue() const
{
  auto usedValues = this->GetUsedLabelValues();
  return usedValues.back() + 1;
}

mitk::Label* mitk::LabelSetImage::AddLabel(mitk::Label* label, GroupIndexType groupID, bool addAsClone, bool correctLabelValue)
{
  if (nullptr == label) mitkThrow() << "Invalid use of AddLabel. label is not valid.";

  mitk::Label::Pointer newLabel = label;

  {
    std::lock_guard<std::shared_mutex> guard(m_LabelNGroupMapsMutex);

    unsigned int max_size = mitk::Label::MAX_LABEL_VALUE + 1;
    if (m_LayerContainer.size() >= max_size)
      return nullptr;

    if (addAsClone) newLabel = label->Clone();

    auto pixelValue = newLabel->GetValue();
    auto usedValues = this->GetUsedLabelValues();
    auto finding = std::find(usedValues.begin(), usedValues.end(), pixelValue);

    if (!usedValues.empty() && usedValues.end() != finding)
    {
      if (correctLabelValue)
      {
        pixelValue = this->GetUnusedLabelValue();
        newLabel->SetValue(pixelValue);
      }
      else
      {
        mitkThrow() << "Cannot add label due to conflicting label value that already exists in the MultiLabelSegmentation. Conflicting label value: " << pixelValue;
      }
    }

    // add DICOM information of the label
    DICOMSegmentationPropertyHelper::SetDICOMSegmentProperties(newLabel);

    this->AddLabelToMap(pixelValue, newLabel, groupID);
    this->RegisterLabel(newLabel);
  }

  this->InvokeEvent(LabelAddedEvent(newLabel->GetValue()));
  m_ActiveLabelValue = newLabel->GetValue();
  this->Modified();

  return newLabel;
}

mitk::Label* mitk::LabelSetImage::AddLabelWithContent(Label* label, const Image* labelContent, GroupIndexType groupID, LabelValueType contentLabelValue, bool addAsClone, bool correctLabelValue)
{
  if (nullptr == labelContent) mitkThrow() << "Invalid use of AddLabel. labelContent is not valid.";
  if (!Equal(*(this->GetTimeGeometry()), *(labelContent->GetTimeGeometry()), mitk::NODE_PREDICATE_GEOMETRY_DEFAULT_CHECK_COORDINATE_PRECISION, mitk::NODE_PREDICATE_GEOMETRY_DEFAULT_CHECK_DIRECTION_PRECISION))
    mitkThrow() << "Invalid use of AddLabel. labelContent has not the same geometry like the segmentation.";

  auto newLabel = this->AddLabel(label, groupID, addAsClone, correctLabelValue);

  mitk::TransferLabelContent(labelContent, this->GetGroupImage(groupID), this->GetConstLabelsByValue(this->GetLabelValuesByGroup(groupID)),
    mitk::LabelSetImage::UNLABELED_VALUE, mitk::LabelSetImage::UNLABELED_VALUE, false, { {contentLabelValue, newLabel->GetValue()}},
    mitk::MultiLabelSegmentation::MergeStyle::Replace, mitk::MultiLabelSegmentation::OverwriteStyle::RegardLocks);

  this->Modified();

  return newLabel;
}

mitk::Label* mitk::LabelSetImage::AddLabel(const std::string& name, const mitk::Color& color, GroupIndexType groupID)
{
  mitk::Label::Pointer newLabel = mitk::Label::New();
  newLabel->SetName(name);
  newLabel->SetColor(color);
  return AddLabel(newLabel,groupID,false);
}

void mitk::LabelSetImage::RenameLabel(LabelValueType pixelValue, const std::string& name, const mitk::Color& color)
{
  std::shared_lock<std::shared_mutex> guard(m_LabelNGroupMapsMutex);

  auto label = GetLabel(pixelValue);
  if (label.IsNull()) mitkThrow() << "Cannot rename label. Unknown label value provided. Unknown label value:" << pixelValue;

  label->SetName(name);
  label->SetColor(color);

  this->UpdateLookupTable(pixelValue);
  m_LookupTable->Modified();

  // change DICOM information of the label
  DICOMSegmentationPropertyHelper::SetDICOMSegmentProperties(label);
}

mitk::Label *mitk::LabelSetImage::GetActiveLabel()
{
  if (m_ActiveLabelValue == UNLABELED_VALUE) return nullptr;

  auto finding = m_LabelMap.find(m_ActiveLabelValue);
  return finding == m_LabelMap.end() ? nullptr : finding->second;
}

const mitk::Label* mitk::LabelSetImage::GetActiveLabel() const
{
  if (m_ActiveLabelValue == UNLABELED_VALUE) return nullptr;

  auto finding = m_LabelMap.find(m_ActiveLabelValue);
  return finding == m_LabelMap.end() ? nullptr : finding->second;
}

void mitk::LabelSetImage::UpdateCenterOfMass(PixelType pixelValue)
{
  if (4 == this->GetDimension())
  {
    AccessFixedDimensionByItk_1(this->GetGroupImage(this->GetGroupIndexOfLabel(pixelValue)), CalculateCenterOfMassProcessing, 4, pixelValue);
  }
  else
  {
    AccessByItk_1(this->GetGroupImage(this->GetGroupIndexOfLabel(pixelValue)), CalculateCenterOfMassProcessing, pixelValue);
  }
}

void mitk::LabelSetImage::SetLookupTable(mitk::LookupTable* lut)
{
  m_LookupTable = lut;
  this->Modified();
}

void mitk::LabelSetImage::UpdateLookupTable(PixelType pixelValue)
{
  auto label = this->GetLabel(pixelValue);
  if (label.IsNull()) mitkThrow() << "Cannot update lookup table. Unknown label value provided. Unknown label value:" << pixelValue;

  const mitk::Color& color = label->GetColor();

  double rgba[4];
  m_LookupTable->GetTableValue(static_cast<int>(pixelValue), rgba);
  rgba[0] = color.GetRed();
  rgba[1] = color.GetGreen();
  rgba[2] = color.GetBlue();
  if (label->GetVisible())
    rgba[3] = label->GetOpacity();
  else
    rgba[3] = 0.0;
  m_LookupTable->SetTableValue(static_cast<int>(pixelValue), rgba);
}

unsigned int mitk::LabelSetImage::GetNumberOfLabels(unsigned int layer) const
{
  if (layer >= m_Groups.size()) mitkThrow() << "Cannot get number of labels in group. Group is unknown. Invalid index:" << layer;
  return m_GroupToLabelMap[layer].size();
}

unsigned int mitk::LabelSetImage::GetTotalNumberOfLabels() const
{
  return m_LabelMap.size();
}

void mitk::LabelSetImage::MaskStamp(mitk::Image *mask, bool forceOverwrite)
{
  try
  {
    mitk::PadImageFilter::Pointer padImageFilter = mitk::PadImageFilter::New();
    padImageFilter->SetInput(0, mask);
    padImageFilter->SetInput(1, this);
    padImageFilter->SetPadConstant(0);
    padImageFilter->SetBinaryFilter(false);
    padImageFilter->SetLowerThreshold(0);
    padImageFilter->SetUpperThreshold(1);

    padImageFilter->Update();

    mitk::Image::Pointer paddedMask = padImageFilter->GetOutput();

    if (paddedMask.IsNull())
      return;

    AccessByItk_2(this, MaskStampProcessing, paddedMask, forceOverwrite);
  }
  catch (...)
  {
    mitkThrow() << "Could not stamp the provided mask on the selected label.";
  }
}

void mitk::LabelSetImage::InitializeByLabeledImage(mitk::Image::Pointer image)
{
  if (image.IsNull() || image->IsEmpty() || !image->IsInitialized())
    mitkThrow() << "Invalid labeled image.";

  try
  {
    this->Initialize(image);

    unsigned int byteSize = sizeof(LabelSetImage::PixelType);
    for (unsigned int dim = 0; dim < image->GetDimension(); ++dim)
    {
      byteSize *= image->GetDimension(dim);
    }

    mitk::ImageWriteAccessor *accessor = new mitk::ImageWriteAccessor(static_cast<mitk::Image *>(this));
    memset(accessor->GetData(), 0, byteSize);
    delete accessor;

    auto geometry = image->GetTimeGeometry()->Clone();
    this->SetTimeGeometry(geometry);

    if (image->GetDimension() == 3)
    {
      AccessTwoImagesFixedDimensionByItk(this, image, InitializeByLabeledImageProcessing, 3);
    }
    else if (image->GetDimension() == 4)
    {
      AccessTwoImagesFixedDimensionByItk(this, image, InitializeByLabeledImageProcessing, 4);
    }
    else
    {
      mitkThrow() << image->GetDimension() << "-dimensional label set images not yet supported";
    }
  }
  catch (Exception& e)
  {
    mitkReThrow(e) << "Could not initialize by provided labeled image.";
  }
  catch (...)
  {
    mitkThrow() << "Could not initialize by provided labeled image due to unknown error.";
  }
  this->Modified();
}

template <typename LabelSetImageType, typename ImageType>
void mitk::LabelSetImage::InitializeByLabeledImageProcessing(LabelSetImageType *labelSetImage, ImageType *image)
{
  typedef itk::ImageRegionConstIteratorWithIndex<ImageType> SourceIteratorType;
  typedef itk::ImageRegionIterator<LabelSetImageType> TargetIteratorType;

  TargetIteratorType targetIter(labelSetImage, labelSetImage->GetRequestedRegion());
  targetIter.GoToBegin();

  SourceIteratorType sourceIter(image, image->GetRequestedRegion());
  sourceIter.GoToBegin();

  while (!sourceIter.IsAtEnd())
  {
    const auto originalSourceValue = sourceIter.Get();
    const auto sourceValue = static_cast<PixelType>(originalSourceValue);

    if (originalSourceValue > mitk::Label::MAX_LABEL_VALUE)
    {
      mitkThrow() << "Cannot initialize MultiLabelSegmentation by image. Image contains a pixel value that exceeds the label value range. Invalid pixel value:" << originalSourceValue;
    }

    targetIter.Set(sourceValue);

    if (LabelSetImage::UNLABELED_VALUE!=sourceValue && !this->ExistLabel(sourceValue))
    {
      if (this->GetTotalNumberOfLabels() >= mitk::Label::MAX_LABEL_VALUE)
      {
        mitkThrow() << "Cannot initialize MultiLabelSegmentation by image. Image contains to many labels.";
      }

      std::stringstream name;
      name << "object-" << sourceValue;

      double rgba[4];
      this->GetLookupTable()->GetTableValue(sourceValue, rgba);

      mitk::Color color;
      color.SetRed(rgba[0]);
      color.SetGreen(rgba[1]);
      color.SetBlue(rgba[2]);

      auto label = mitk::Label::New();
      label->SetName(name.str().c_str());
      label->SetColor(color);
      label->SetOpacity(rgba[3]);
      label->SetValue(sourceValue);

      this->AddLabel(label,0,false);
    }

    ++sourceIter;
    ++targetIter;
  }
}

template <typename ImageType>
void mitk::LabelSetImage::MaskStampProcessing(ImageType *itkImage, mitk::Image *mask, bool forceOverwrite)
{
  typename ImageType::Pointer itkMask;
  mitk::CastToItkImage(mask, itkMask);

  typedef itk::ImageRegionConstIterator<ImageType> SourceIteratorType;
  typedef itk::ImageRegionIterator<ImageType> TargetIteratorType;

  SourceIteratorType sourceIter(itkMask, itkMask->GetLargestPossibleRegion());
  sourceIter.GoToBegin();

  TargetIteratorType targetIter(itkImage, itkImage->GetLargestPossibleRegion());
  targetIter.GoToBegin();

  const auto activeLabel = this->GetActiveLabel()->GetValue();

  while (!sourceIter.IsAtEnd())
  {
    PixelType sourceValue = sourceIter.Get();
    PixelType targetValue = targetIter.Get();

    if ((sourceValue != UNLABELED_VALUE) &&
        (forceOverwrite || !this->IsLabelLocked(targetValue))) // skip unlabeled pixels and locked labels
    {
      targetIter.Set(activeLabel);
    }
    ++sourceIter;
    ++targetIter;
  }

  this->Modified();
}

template <typename ImageType>
void mitk::LabelSetImage::CalculateCenterOfMassProcessing(ImageType *itkImage, LabelValueType pixelValue)
{
  if (ImageType::GetImageDimension() != 3)
  {
    return;
  }

  auto labelGeometryFilter = itk::LabelGeometryImageFilter<ImageType>::New();
  labelGeometryFilter->SetInput(itkImage);
  labelGeometryFilter->Update();
  auto centroid = labelGeometryFilter->GetCentroid(pixelValue);

  mitk::Point3D pos;
  pos[0] = centroid[0];
  pos[1] = centroid[1];
  pos[2] = centroid[2];

  auto label = this->GetLabel(pixelValue);
  if (label.IsNotNull())
  {
    label->SetCenterOfMassIndex(pos);
    this->GetSlicedGeometry()->IndexToWorld(pos, pos);
    label->SetCenterOfMassCoordinates(pos);
  }
}

template <typename TPixel, unsigned int VImageDimension>
void mitk::LabelSetImage::LayerContainerToImageProcessing(itk::Image<TPixel, VImageDimension> *target,
                                                          unsigned int layer)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typename ImageType::Pointer itkSource;
  // mitk::CastToItkImage(m_LayerContainer[layer], itkSource);
  itkSource = ImageToItkImage<TPixel, VImageDimension>(m_LayerContainer[layer]);
  typedef itk::ImageRegionConstIterator<ImageType> SourceIteratorType;
  typedef itk::ImageRegionIterator<ImageType> TargetIteratorType;

  SourceIteratorType sourceIter(itkSource, itkSource->GetLargestPossibleRegion());
  sourceIter.GoToBegin();

  TargetIteratorType targetIter(target, target->GetLargestPossibleRegion());
  targetIter.GoToBegin();

  while (!sourceIter.IsAtEnd())
  {
    targetIter.Set(sourceIter.Get());
    ++sourceIter;
    ++targetIter;
  }
}

template <typename TPixel, unsigned int VImageDimension>
void mitk::LabelSetImage::ImageToLayerContainerProcessing(const itk::Image<TPixel, VImageDimension> *source,
                                                          unsigned int layer) const
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typename ImageType::Pointer itkTarget;
  // mitk::CastToItkImage(m_LayerContainer[layer], itkTarget);
  itkTarget = ImageToItkImage<TPixel, VImageDimension>(m_LayerContainer[layer]);

  typedef itk::ImageRegionConstIterator<ImageType> SourceIteratorType;
  typedef itk::ImageRegionIterator<ImageType> TargetIteratorType;

  SourceIteratorType sourceIter(source, source->GetLargestPossibleRegion());
  sourceIter.GoToBegin();

  TargetIteratorType targetIter(itkTarget, itkTarget->GetLargestPossibleRegion());
  targetIter.GoToBegin();

  while (!sourceIter.IsAtEnd())
  {
    targetIter.Set(sourceIter.Get());
    ++sourceIter;
    ++targetIter;
  }

  m_LayerContainer[layer]->Modified();
}

template <typename ImageType>
void mitk::LabelSetImage::EraseLabelProcessing(ImageType *itkImage, PixelType pixelValue)
{
  typedef itk::ImageRegionIterator<ImageType> IteratorType;

  IteratorType iter(itkImage, itkImage->GetLargestPossibleRegion());
  iter.GoToBegin();

  while (!iter.IsAtEnd())
  {
    PixelType value = iter.Get();

    if (value == pixelValue)
    {
      iter.Set(0);
    }
    ++iter;
  }
}

template <typename ImageType>
void mitk::LabelSetImage::MergeLabelProcessing(ImageType *itkImage, PixelType pixelValue, PixelType index)
{
  typedef itk::ImageRegionIterator<ImageType> IteratorType;

  IteratorType iter(itkImage, itkImage->GetLargestPossibleRegion());
  iter.GoToBegin();

  while (!iter.IsAtEnd())
  {
    if (iter.Get() == index)
    {
      iter.Set(pixelValue);
    }
    ++iter;
  }
}

void mitk::LabelSetImage::AddLabelToMap(LabelValueType labelValue, mitk::Label* label, GroupIndexType groupID)
{
  if (m_LabelMap.find(labelValue)!=m_LabelMap.end())
    mitkThrow() << "Segmentation is in an invalid state: Label value collision. A label was added with a LabelValue already in use. LabelValue: " << labelValue;

  if (!this->ExistGroup(groupID))
    mitkThrow() << "Cannot add label. Defined group is unknown. Invalid group index: " << groupID;

  m_LabelMap[labelValue] = label;
  m_LabelToGroupMap[labelValue] = groupID;
  auto groupFinding = std::find(m_GroupToLabelMap[groupID].begin(), m_GroupToLabelMap[groupID].end(), labelValue);
  if (groupFinding == m_GroupToLabelMap[groupID].end())
  {
    m_GroupToLabelMap[groupID].push_back(labelValue);
  }
}

void mitk::LabelSetImage::RegisterLabel(mitk::Label* label)
{
  if (nullptr == label) mitkThrow() << "Invalid call of RegisterLabel with a nullptr.";

  UpdateLookupTable(label->GetValue());
  m_LookupTable->Modified();

  auto command = itk::MemberCommand<LabelSetImage>::New();
  command->SetCallbackFunction(this, &LabelSetImage::OnLabelModified);
  m_LabelModEventGuardMap.emplace(label->GetValue(), ITKEventObserverGuard(label, itk::ModifiedEvent(), command));
}

void mitk::LabelSetImage::ReleaseLabel(Label* label)
{
  if (nullptr == label) mitkThrow() << "Invalid call of ReleaseLabel with a nullptr.";
  m_LabelModEventGuardMap.erase(label->GetValue());
}

void mitk::LabelSetImage::ApplyToLabels(const LabelValueVectorType& values, std::function<void(Label*)>&& lambda)
{
  auto labels = this->GetLabelsByValue(values);
  std::for_each(labels.begin(), labels.end(), lambda);
  this->InvokeEvent(LabelsChangedEvent(values));
}

void mitk::LabelSetImage::VisitLabels(const LabelValueVectorType& values, std::function<void(const Label*)>&& lambda) const
{
  auto labels = this->GetConstLabelsByValue(values);
  std::for_each(labels.begin(), labels.end(), lambda);
}


void mitk::LabelSetImage::OnLabelModified(const Object* sender, const itk::EventObject&)
{
  auto label = dynamic_cast<const Label*>(sender);
  if (nullptr == label)
    mitkThrow() << "LabelSet is in wrong state. LabelModified event is not send by a label instance.";

  Superclass::Modified();
  this->InvokeEvent(LabelModifiedEvent(label->GetValue()));
}

bool mitk::LabelSetImage::ExistLabel(LabelValueType value) const
{
  auto finding = m_LabelMap.find(value);
  return m_LabelMap.end() != finding;
}

bool mitk::LabelSetImage::ExistLabel(LabelValueType value, GroupIndexType groupIndex) const
{
  auto finding = m_LabelToGroupMap.find(value);
  if (m_LabelToGroupMap.end() != finding)
  {
    return finding->second == groupIndex;
  }
  return false;
}

bool mitk::LabelSetImage::ExistGroup(GroupIndexType index) const
{
  return index < m_LayerContainer.size();
}

mitk::LabelSetImage::GroupIndexType mitk::LabelSetImage::GetGroupIndexOfLabel(LabelValueType value) const
{
  auto finding = m_LabelToGroupMap.find(value);
  if (m_LabelToGroupMap.end() == finding)
  {
    mitkThrow()<< "Cannot deduce group index. Passed label value does not exist. Value: "<< value;
  }
  return finding->second;
}


mitk::Label::ConstPointer mitk::LabelSetImage::GetLabel(LabelValueType value) const
{
  auto finding = m_LabelMap.find(value);
  if (m_LabelMap.end() != finding)
  {
    return finding->second;
  }
  return nullptr;
};

mitk::Label::Pointer mitk::LabelSetImage::GetLabel(LabelValueType value)
{
  auto finding = m_LabelMap.find(value);
  if (m_LabelMap.end() != finding)
  {
    return finding->second;
  }
  return nullptr;
};

bool mitk::LabelSetImage::IsLabelLocked(LabelValueType value) const
{
  if (value == UNLABELED_VALUE)
  {
    return m_UnlabeledLabelLock;
  }

  const auto label = this->GetLabel(value);
  return label->GetLocked();
}

const mitk::LabelSetImage::ConstLabelVectorType mitk::LabelSetImage::GetLabels() const
{
  ConstLabelVectorType result;
  for (auto [value, label] : m_LabelMap)
  {
    (void)value; // Prevent unused variable error in older compilers
    result.emplace_back(label);
  }
  return result;
}

const mitk::LabelSetImage::LabelVectorType mitk::LabelSetImage::GetLabels()
{
  LabelVectorType result;
  for (auto [value, label] : m_LabelMap)
  {
    (void)value; // Prevent unused variable error in older compilers
    result.emplace_back(label);
  }
  return result;
}

const mitk::LabelSetImage::LabelVectorType mitk::LabelSetImage::GetLabelsByValue(const LabelValueVectorType& labelValues, bool ignoreMissing)
{
  LabelVectorType result;
  for (const auto& labelValue : labelValues)
  {
    Label::Pointer label = this->GetLabel(labelValue);

    if (label.IsNotNull())
    {
      result.emplace_back(label);
    }
    else if (!ignoreMissing) mitkThrow() << "Error cannot get labels by Value. At least one passed value is unknown. Unknown value: " << labelValue;
  }
  return result;
}

const mitk::LabelSetImage::ConstLabelVectorType mitk::LabelSetImage::GetConstLabelsByValue(const LabelValueVectorType& labelValues, bool ignoreMissing) const
{
  ConstLabelVectorType result;
  for (const auto& labelValue : labelValues)
  {
    Label::ConstPointer label = this->GetLabel(labelValue);

    if (label.IsNotNull())
    {
      result.emplace_back(label);
    }
    else if (!ignoreMissing) mitkThrow() << "Error cannot get labels by Value. At least one passed value is unknown. Unknown value: " << labelValue;
  }
  return result;
}

const mitk::LabelSetImage::LabelValueVectorType mitk::LabelSetImage::GetLabelValuesByGroup(GroupIndexType index) const
{
  if (!this->ExistGroup(index))
    mitkThrow() << "Cannot get labels of an invalid group. Invalid group index: " << index;

  return m_GroupToLabelMap[index];
}

const mitk::LabelSetImage::LabelValueVectorType mitk::LabelSetImage::GetLabelValuesByName(GroupIndexType index, const std::string_view name) const
{
  LabelValueVectorType result;

  auto searchName = [&result, name](const Label* l) { if(l->GetName() == name) result.push_back(l->GetValue()); };

  this->VisitLabels(this->GetLabelValuesByGroup(index), searchName);

  return result;
}

std::vector<std::string> mitk::LabelSetImage::GetLabelClassNames() const
{
  std::set<std::string> names;
  auto searchName = [&names](const Label* l) { names.emplace(l->GetName()); };
  this->VisitLabels(this->GetAllLabelValues(), searchName);

  return std::vector<std::string>(names.begin(), names.end());
}

std::vector<std::string> mitk::LabelSetImage::GetLabelClassNamesByGroup(GroupIndexType index) const
{
  std::set<std::string> names;
  auto searchName = [&names](const Label* l) { names.emplace(l->GetName()); };
  this->VisitLabels(this->GetLabelValuesByGroup(index), searchName);

  return std::vector<std::string>(names.begin(), names.end());
}

void mitk::LabelSetImage::SetAllLabelsVisible(bool visible)
{
  auto setVisibility = [visible,this](Label* l)
    {
      l->SetVisible(visible);
      this->UpdateLookupTable(l->GetValue());
    };

  this->ApplyToLabels(this->GetAllLabelValues(), setVisibility);
  this->m_LookupTable->Modified();
}

void mitk::LabelSetImage::SetAllLabelsVisibleByGroup(GroupIndexType group, bool visible)
{
  auto setVisibility = [visible, this](Label* l)
    {
      l->SetVisible(visible);
      this->UpdateLookupTable(l->GetValue());
    };

  this->ApplyToLabels(this->GetLabelValuesByGroup(group), setVisibility);
  this->m_LookupTable->Modified();
}

void mitk::LabelSetImage::SetAllLabelsVisibleByName(GroupIndexType group, const std::string_view name, bool visible)
{
  auto setVisibility = [visible, this](Label* l)
    {
      l->SetVisible(visible);
      this->UpdateLookupTable(l->GetValue());
    };

  this->ApplyToLabels(this->GetLabelValuesByName(group, name), setVisibility);
  this->m_LookupTable->Modified();
}

void mitk::LabelSetImage::SetAllLabelsLocked(bool locked)
{
  auto setLock = [locked](Label* l) { l->SetLocked(locked); };

  this->ApplyToLabels(this->GetAllLabelValues(), setLock);
}

void mitk::LabelSetImage::SetAllLabelsLockedByGroup(GroupIndexType group, bool locked)
{
  auto setLock = [locked](Label* l) { l->SetLocked(locked); };

  this->ApplyToLabels(this->GetLabelValuesByGroup(group), setLock);
}

void mitk::LabelSetImage::SetAllLabelsLockedByName(GroupIndexType group, const std::string_view name, bool locked)
{
  auto setLock = [locked](Label* l) { l->SetLocked(locked); };

  this->ApplyToLabels(this->GetLabelValuesByName(group, name), setLock);
}

bool mitk::Equal(const mitk::LabelSetImage &leftHandSide,
                 const mitk::LabelSetImage &rightHandSide,
                 ScalarType eps,
                 bool verbose)
{
  bool returnValue = true;

  /* LabelSetImage members */

  MITK_INFO(verbose) << "--- LabelSetImage Equal ---";

  // m_LookupTable;
  const mitk::LookupTable* lhsLUT = leftHandSide.GetLookupTable();
  const mitk::LookupTable* rhsLUT = rightHandSide.GetLookupTable();

  returnValue = *lhsLUT == *rhsLUT;
  if (!returnValue)
  {
    MITK_INFO(verbose) << "Lookup tables not equal.";
    return returnValue;
    ;
  }

  // number layers
  returnValue = leftHandSide.GetNumberOfLayers() == rightHandSide.GetNumberOfLayers();
  if (!returnValue)
  {
    MITK_INFO(verbose) << "Number of layers not equal.";
    return false;
  }

  // total number labels
  returnValue = leftHandSide.GetTotalNumberOfLabels() == rightHandSide.GetTotalNumberOfLabels();
  if (!returnValue)
  {
    MITK_INFO(verbose) << "Total number of labels not equal.";
    return false;
  }

  // active layer
  returnValue = leftHandSide.GetActiveLayer() == rightHandSide.GetActiveLayer();
  if (!returnValue)
  {
    MITK_INFO(verbose) << "Active layer not equal.";
    return false;
  }

  if (4 == leftHandSide.GetDimension())
  {
    MITK_INFO(verbose) << "Can not compare image data for 4D images - skipping check.";
  }
  else
  {
    // working image data
    returnValue = mitk::Equal((const mitk::Image &)leftHandSide, (const mitk::Image &)rightHandSide, eps, verbose);
    if (!returnValue)
    {
      MITK_INFO(verbose) << "Working image data not equal.";
      return false;
    }
  }

  if (leftHandSide.GetTotalNumberOfLabels() != rightHandSide.GetTotalNumberOfLabels())
  {
    MITK_INFO(verbose) << "Number of labels are not equal.";
    return false;
  }

  for (unsigned int layerIndex = 0; layerIndex < leftHandSide.GetNumberOfLayers(); layerIndex++)
  {
    if (4 == leftHandSide.GetDimension())
    {
      MITK_INFO(verbose) << "Can not compare image data for 4D images - skipping check.";
    }
    else
    {
      // layer image data
      returnValue =
        mitk::Equal(*leftHandSide.GetGroupImage(layerIndex), *rightHandSide.GetGroupImage(layerIndex), eps, verbose);
      if (!returnValue)
      {
        MITK_INFO(verbose) << "Layer image data not equal.";
        return false;
      }
    }

    // label data
    auto leftLabelsInGroup = leftHandSide.GetLabelValuesByGroup(layerIndex);
    auto rightLabelsInGroup = rightHandSide.GetLabelValuesByGroup(layerIndex);

    if (leftLabelsInGroup.size()!=rightLabelsInGroup.size())
    {
      MITK_INFO(verbose) << "Number of layer labels is not equal. Invalid layer:" <<layerIndex;
      return false;
    }

    for (ConstLabelVector::size_type index = 0; index < leftLabelsInGroup.size(); ++index)
    {
      if (!mitk::Equal(*(leftHandSide.GetLabel(leftLabelsInGroup[index])), *(rightHandSide.GetLabel(rightLabelsInGroup[index])),eps,verbose))
      {
        MITK_INFO(verbose) << "At least one label in layer is not equal. Invalid layer:" << layerIndex;
        return false;
      }
    }
  }

  return returnValue;
}

bool mitk::Equal(const mitk::LabelSetImage::ConstLabelVectorType& leftHandSide,
  const mitk::LabelSetImage::ConstLabelVectorType& rightHandSide, ScalarType eps, bool verbose)
{
  bool returnValue = true;

  // container size;
  returnValue = leftHandSide.size() == rightHandSide.size();
  if (!returnValue)
  {
    MITK_INFO(verbose) << "Number of labels not equal.";
    return returnValue;
    ;
  }

  // m_LabelContainer;
  auto lhsit = leftHandSide.begin();
  auto rhsit = rightHandSide.begin();
  for (; lhsit != leftHandSide.end(); ++lhsit, ++rhsit)
  {
    returnValue = mitk::Equal(**rhsit, **lhsit,eps,verbose);
    if (!returnValue)
    {
      MITK_INFO(verbose) << "Label in label container not equal.";
      return returnValue;
      ;
    }
  }

  return returnValue;
}

/**Helper function to convert a vector of labels into a label map
 * @pre every label in the vector has a unique value.*/
using ConstLabelMapType = std::map<mitk::LabelSetImage::LabelValueType, mitk::Label::ConstPointer>;
ConstLabelMapType ConvertLabelVectorToMap(const mitk::ConstLabelVector& labelV)
{
  ConstLabelMapType result;
  for (auto label : labelV)
  {
    const auto value = label->GetValue();
    auto finding = result.find(value);
    if (finding != result.end()) mitkThrow() << "Operation failed. Cannot convert label vector into label map, because at least one label value is not unique. Violating label value: " << value;
    result.insert(std::make_pair(value, label));
  }

  return result;
}


/** Functor class that implements the label transfer and is used in conjunction with the itk::BinaryFunctorImageFilter.
* For details regarding the usage of the filter and the functor patterns, please see info of itk::BinaryFunctorImageFilter.
*/
template <class TDestinationPixel, class TSourcePixel, class TOutputpixel>
class LabelTransferFunctor
{

public:
  LabelTransferFunctor() {};

  LabelTransferFunctor(const ConstLabelMapType& destinationLabels, mitk::Label::PixelType sourceBackground,
    mitk::Label::PixelType destinationBackground, bool destinationBackgroundLocked,
    mitk::Label::PixelType sourceLabel, mitk::Label::PixelType newDestinationLabel, mitk::MultiLabelSegmentation::MergeStyle mergeStyle,
    mitk::MultiLabelSegmentation::OverwriteStyle overwriteStyle) :
    m_DestinationLabels(destinationLabels), m_SourceBackground(sourceBackground),
    m_DestinationBackground(destinationBackground), m_DestinationBackgroundLocked(destinationBackgroundLocked),
    m_SourceLabel(sourceLabel), m_NewDestinationLabel(newDestinationLabel), m_MergeStyle(mergeStyle), m_OverwriteStyle(overwriteStyle)
  {
  };

  ~LabelTransferFunctor() {};

  bool operator!=(const LabelTransferFunctor& other)const
  {
    return !(*this == other);
  }
  bool operator==(const LabelTransferFunctor& other) const
  {
    return this->m_SourceBackground == other.m_SourceBackground &&
      this->m_DestinationBackground == other.m_DestinationBackground &&
      this->m_DestinationBackgroundLocked == other.m_DestinationBackgroundLocked &&
      this->m_SourceLabel == other.m_SourceLabel &&
      this->m_NewDestinationLabel == other.m_NewDestinationLabel &&
      this->m_MergeStyle == other.m_MergeStyle &&
      this->m_OverwriteStyle == other.m_OverwriteStyle &&
      this->m_DestinationLabels == other.m_DestinationLabels;
  }

  LabelTransferFunctor& operator=(const LabelTransferFunctor& other)
  {
    this->m_DestinationLabels = other.m_DestinationLabels;
    this->m_SourceBackground = other.m_SourceBackground;
    this->m_DestinationBackground = other.m_DestinationBackground;
    this->m_DestinationBackgroundLocked = other.m_DestinationBackgroundLocked;
    this->m_SourceLabel = other.m_SourceLabel;
    this->m_NewDestinationLabel = other.m_NewDestinationLabel;
    this->m_MergeStyle = other.m_MergeStyle;
    this->m_OverwriteStyle = other.m_OverwriteStyle;

    return *this;
  }

  inline TOutputpixel operator()(const TDestinationPixel& existingDestinationValue, const TSourcePixel& existingSourceValue)
  {
    if (existingSourceValue == this->m_SourceLabel)
    {
      if (mitk::MultiLabelSegmentation::OverwriteStyle::IgnoreLocks == this->m_OverwriteStyle)
      {
        return this->m_NewDestinationLabel;
      }
      else
      {
        if (existingDestinationValue == m_DestinationBackground)
        {
          if (!m_DestinationBackgroundLocked)
          {
            return this->m_NewDestinationLabel;
          }
        }
        else
        {
          auto labelFinding = this->m_DestinationLabels.find(existingDestinationValue);
          if (labelFinding==this->m_DestinationLabels.end() || !labelFinding->second->GetLocked())
          {
            return this->m_NewDestinationLabel;
          }
        }
      }
    }
    else if (mitk::MultiLabelSegmentation::MergeStyle::Replace == this->m_MergeStyle
      && existingSourceValue == this->m_SourceBackground
      && existingDestinationValue == this->m_NewDestinationLabel
      && (mitk::MultiLabelSegmentation::OverwriteStyle::IgnoreLocks == this->m_OverwriteStyle
          || !this->m_DestinationBackgroundLocked))
    {
      return this->m_DestinationBackground;
    }

    return existingDestinationValue;
  }

private:
  ConstLabelMapType m_DestinationLabels;
  mitk::Label::PixelType m_SourceBackground = 0;
  mitk::Label::PixelType m_DestinationBackground = 0;
  bool m_DestinationBackgroundLocked = false;
  mitk::Label::PixelType m_SourceLabel = 1;
  mitk::Label::PixelType m_NewDestinationLabel = 1;
  mitk::MultiLabelSegmentation::MergeStyle m_MergeStyle = mitk::MultiLabelSegmentation::MergeStyle::Replace;
  mitk::MultiLabelSegmentation::OverwriteStyle m_OverwriteStyle = mitk::MultiLabelSegmentation::OverwriteStyle::RegardLocks;
};

/**Helper function used by TransferLabelContentAtTimeStep to allow the templating over different image dimensions in conjunction of AccessFixedPixelTypeByItk_n.*/
template<unsigned int VImageDimension>
void TransferLabelContentAtTimeStepHelper(const itk::Image<mitk::Label::PixelType, VImageDimension>* itkSourceImage, mitk::Image* destinationImage,
  const mitk::ConstLabelVector& destinationLabels, mitk::Label::PixelType sourceBackground, mitk::Label::PixelType destinationBackground,
  bool destinationBackgroundLocked, mitk::Label::PixelType sourceLabel, mitk::Label::PixelType newDestinationLabel, mitk::MultiLabelSegmentation::MergeStyle mergeStyle, mitk::MultiLabelSegmentation::OverwriteStyle overwriteStyle)
{
  typedef itk::Image<mitk::Label::PixelType, VImageDimension> ContentImageType;
  typename ContentImageType::Pointer itkDestinationImage;
  mitk::CastToItkImage(destinationImage, itkDestinationImage);

  auto sourceRegion = itkSourceImage->GetLargestPossibleRegion();
  auto relevantRegion = itkDestinationImage->GetLargestPossibleRegion();
  bool overlapping = relevantRegion.Crop(sourceRegion);

  if (!overlapping)
  {
    mitkThrow() << "Invalid call of TransferLabelContentAtTimeStep; sourceImage and destinationImage seem to have no overlapping image region.";
  }

  typedef LabelTransferFunctor <mitk::Label::PixelType, mitk::Label::PixelType, mitk::Label::PixelType> LabelTransferFunctorType;
  typedef itk::BinaryFunctorImageFilter<ContentImageType, ContentImageType, ContentImageType, LabelTransferFunctorType> FilterType;

  LabelTransferFunctorType transferFunctor(ConvertLabelVectorToMap(destinationLabels), sourceBackground, destinationBackground,
    destinationBackgroundLocked, sourceLabel, newDestinationLabel, mergeStyle, overwriteStyle);

  auto transferFilter = FilterType::New();

  transferFilter->SetFunctor(transferFunctor);
  transferFilter->InPlaceOn();
  transferFilter->SetInput1(itkDestinationImage);
  transferFilter->SetInput2(itkSourceImage);
  transferFilter->GetOutput()->SetRequestedRegion(relevantRegion);

  transferFilter->Update();
}

void mitk::TransferLabelContentAtTimeStep(
  const Image* sourceImage, Image* destinationImage, const mitk::ConstLabelVector& destinationLabels, const TimeStepType timeStep, mitk::Label::PixelType sourceBackground,
  mitk::Label::PixelType destinationBackground, bool destinationBackgroundLocked, LabelValueMappingVector labelMapping,
  MultiLabelSegmentation::MergeStyle mergeStyle, MultiLabelSegmentation::OverwriteStyle overwriteStlye)
{
  if (nullptr == sourceImage)
  {
    mitkThrow() << "Invalid call of TransferLabelContentAtTimeStep; sourceImage must not be null.";
  }
  if (nullptr == destinationImage)
  {
    mitkThrow() << "Invalid call of TransferLabelContentAtTimeStep; destinationImage must not be null.";
  }

  if (sourceImage == destinationImage && labelMapping.size() > 1)
  {
    MITK_DEBUG << "Warning. Using TransferLabelContentAtTimeStep or TransferLabelContent with equal source and destination and more then on label to transfer, can lead to wrong results. Please see documentation and verify that the usage is OK.";
  }

  Image::ConstPointer sourceImageAtTimeStep = SelectImageByTimeStep(sourceImage, timeStep);
  Image::Pointer destinationImageAtTimeStep = SelectImageByTimeStep(destinationImage, timeStep);

  if (nullptr == sourceImageAtTimeStep)
  {
    mitkThrow() << "Invalid call of TransferLabelContentAtTimeStep; sourceImage does not have the requested time step: " << timeStep;
  }

  if (nullptr == destinationImageAtTimeStep)
  {
    mitkThrow() << "Invalid call of TransferLabelContentAtTimeStep; destinationImage does not have the requested time step: " << timeStep;
  }

  if (!Equal(*(sourceImageAtTimeStep->GetGeometry()), *(destinationImageAtTimeStep->GetGeometry()), mitk::NODE_PREDICATE_GEOMETRY_DEFAULT_CHECK_COORDINATE_PRECISION, mitk::NODE_PREDICATE_GEOMETRY_DEFAULT_CHECK_DIRECTION_PRECISION))
  {
    if (IsSubGeometry(*(sourceImageAtTimeStep->GetGeometry()), *(destinationImageAtTimeStep->GetGeometry()), mitk::NODE_PREDICATE_GEOMETRY_DEFAULT_CHECK_COORDINATE_PRECISION, mitk::NODE_PREDICATE_GEOMETRY_DEFAULT_CHECK_DIRECTION_PRECISION, true))
    {
      //we have to pad the source image
      //because ImageToImageFilters always check for origin matching even if
      //the requested output region is fitting :(
      auto padFilter = mitk::PadImageFilter::New();
      padFilter->SetInput(0, sourceImageAtTimeStep);
      padFilter->SetInput(1, destinationImageAtTimeStep);
      padFilter->SetPadConstant(Label::UNLABELED_VALUE);
      padFilter->SetBinaryFilter(false);

      padFilter->Update();

      sourceImageAtTimeStep = padFilter->GetOutput();
    }
    else
    {
      mitkThrow() << "Invalid call of TransferLabelContentAtTimeStep; source image has neither the same geometry than destination image nor has the source image a sub geometry.";
    }
  }

  auto destLabelMap = ConvertLabelVectorToMap(destinationLabels);
  for (const auto& [sourceLabel, newDestinationLabel] : labelMapping)
  {
    if (LabelSetImage::UNLABELED_VALUE!=newDestinationLabel && destLabelMap.end() == destLabelMap.find(newDestinationLabel))
    {
      mitkThrow() << "Invalid call of TransferLabelContentAtTimeStep. Defined destination label does not exist in destinationImage. newDestinationLabel: " << newDestinationLabel;
    }

    AccessFixedPixelTypeByItk_n(sourceImageAtTimeStep, TransferLabelContentAtTimeStepHelper, (Label::PixelType), (destinationImageAtTimeStep, destinationLabels, sourceBackground, destinationBackground, destinationBackgroundLocked, sourceLabel, newDestinationLabel, mergeStyle, overwriteStlye));
  }
  destinationImage->Modified();
}

void mitk::TransferLabelContent(
  const Image* sourceImage, Image* destinationImage, const mitk::ConstLabelVector& destinationLabels, mitk::Label::PixelType sourceBackground,
  mitk::Label::PixelType destinationBackground, bool destinationBackgroundLocked, LabelValueMappingVector labelMapping,
  MultiLabelSegmentation::MergeStyle mergeStyle, MultiLabelSegmentation::OverwriteStyle overwriteStlye)
{
  if (nullptr == sourceImage)
  {
    mitkThrow() << "Invalid call of TransferLabelContent; sourceImage must not be null.";
  }
  if (nullptr == destinationImage)
  {
    mitkThrow() << "Invalid call of TransferLabelContent; destinationImage must not be null.";
  }

  const auto sourceTimeStepCount = sourceImage->GetTimeGeometry()->CountTimeSteps();
  if (sourceTimeStepCount != destinationImage->GetTimeGeometry()->CountTimeSteps())
  {
    mitkThrow() << "Invalid call of TransferLabelContent; mismatch between images in number of time steps.";
  }

  for (mitk::TimeStepType i = 0; i < sourceTimeStepCount; ++i)
  {
    TransferLabelContentAtTimeStep(sourceImage, destinationImage, destinationLabels, i, sourceBackground,
      destinationBackground, destinationBackgroundLocked, labelMapping, mergeStyle, overwriteStlye);
  }
}

void mitk::TransferLabelContentAtTimeStep(
  const LabelSetImage* sourceImage, LabelSetImage* destinationImage, const TimeStepType timeStep,
  LabelValueMappingVector labelMapping,
  MultiLabelSegmentation::MergeStyle mergeStyle, MultiLabelSegmentation::OverwriteStyle overwriteStlye)
{
  if (nullptr == sourceImage)
  {
    mitkThrow() << "Invalid call of TransferLabelContentAtTimeStep; sourceImage must not be null.";
  }

  auto destinationLabels = destinationImage->GetConstLabelsByValue(destinationImage->GetLabelValuesByGroup(destinationImage->GetActiveLayer()));

  for (const auto& mappingElement : labelMapping)
  {
    if (LabelSetImage::UNLABELED_VALUE != mappingElement.first && !sourceImage->ExistLabel(mappingElement.first, sourceImage->GetActiveLayer()))
    {
      mitkThrow() << "Invalid call of TransferLabelContentAtTimeStep. Defined source label does not exist in sourceImage. SourceLabel: " << mappingElement.first;
    }
  }

  TransferLabelContentAtTimeStep(sourceImage, destinationImage, destinationLabels, timeStep, LabelSetImage::UNLABELED_VALUE, LabelSetImage::UNLABELED_VALUE, destinationImage->GetUnlabeledLabelLock(),
    labelMapping, mergeStyle, overwriteStlye);
}

void mitk::TransferLabelContent(
  const LabelSetImage* sourceImage, LabelSetImage* destinationImage,
  LabelValueMappingVector labelMapping,
  MultiLabelSegmentation::MergeStyle mergeStyle, MultiLabelSegmentation::OverwriteStyle overwriteStlye)
{
  if (nullptr == sourceImage)
  {
    mitkThrow() << "Invalid call of TransferLabelContent; sourceImage must not be null.";
  }
  if (nullptr == destinationImage)
  {
    mitkThrow() << "Invalid call of TransferLabelContent; destinationImage must not be null.";
  }

  const auto sourceTimeStepCount = sourceImage->GetTimeGeometry()->CountTimeSteps();
  if (sourceTimeStepCount != destinationImage->GetTimeGeometry()->CountTimeSteps())
  {
    mitkThrow() << "Invalid call of TransferLabelContent; images have no equal number of time steps.";
  }

  for (mitk::TimeStepType i = 0; i < sourceTimeStepCount; ++i)
  {
    TransferLabelContentAtTimeStep(sourceImage, destinationImage, i, labelMapping, mergeStyle, overwriteStlye);
  }
}

