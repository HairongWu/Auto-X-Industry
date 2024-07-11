/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

// MITK
#include "mitkTotalSegmentatorTool.h"

#include <mitkIOUtil.h>
#include <mitkImageReadAccessor.h>

#include <algorithm>
#include <mitkFileSystem.h>
#include <itksys/SystemTools.hxx>
#include <regex>

// us
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleContext.h>
#include <usModuleResource.h>
#include <usServiceReference.h>

namespace mitk
{
  MITK_TOOL_MACRO(MITKSEGMENTATION_EXPORT, TotalSegmentatorTool, "Total Segmentator");
}

mitk::TotalSegmentatorTool::~TotalSegmentatorTool()
{
  fs::remove_all(this->GetMitkTempDir());
}

mitk::TotalSegmentatorTool::TotalSegmentatorTool() : SegWithPreviewTool(true) // prevents auto-compute across all timesteps
{
  this->IsTimePointChangeAwareOff();
  this->RequestDeactivationConfirmationOn();
}

void mitk::TotalSegmentatorTool::Activated()
{
  Superclass::Activated();
  this->SetLabelTransferScope(LabelTransferScope::AllLabels);
  this->SetLabelTransferMode(LabelTransferMode::AddLabel);
}

const char **mitk::TotalSegmentatorTool::GetXPM() const
{
  return nullptr;
}

us::ModuleResource mitk::TotalSegmentatorTool::GetIconResource() const
{
  us::Module *module = us::GetModuleContext()->GetModule();
  us::ModuleResource resource = module->GetResource("AI.svg");
  return resource;
}

const char *mitk::TotalSegmentatorTool::GetName() const
{
  return "TotalSegmentator";
}

void mitk::TotalSegmentatorTool::onPythonProcessEvent(itk::Object * /*pCaller*/, const itk::EventObject &e, void *)
{
  std::string testCOUT;
  std::string testCERR;
  const auto *pEvent = dynamic_cast<const mitk::ExternalProcessStdOutEvent *>(&e);

  if (pEvent)
  {
    testCOUT = testCOUT + pEvent->GetOutput();
    MITK_INFO << testCOUT;
  }

  const auto *pErrEvent = dynamic_cast<const mitk::ExternalProcessStdErrEvent *>(&e);

  if (pErrEvent)
  {
    testCERR = testCERR + pErrEvent->GetOutput();
    MITK_ERROR << testCERR;
  }
}

void mitk::TotalSegmentatorTool::DoUpdatePreview(const Image *inputAtTimeStep,
                                                 const Image * /*oldSegAtTimeStep*/,
                                                 LabelSetImage *previewImage,
                                                 TimeStepType timeStep)
{
  if (this->m_MitkTempDir.empty())
  {
    this->SetMitkTempDir(IOUtil::CreateTemporaryDirectory("mitk-XXXXXX"));
  }
  ProcessExecutor::Pointer spExec = ProcessExecutor::New();
  itk::CStyleCommand::Pointer spCommand = itk::CStyleCommand::New();
  spCommand->SetCallback(&onPythonProcessEvent);
  spExec->AddObserver(ExternalProcessOutputEvent(), spCommand);
  m_ProgressCommand->SetProgress(5);

  std::string inDir, outDir, inputImagePath, outputImagePath, scriptPath;
  inDir = IOUtil::CreateTemporaryDirectory("totalseg-in-XXXXXX", this->GetMitkTempDir());
  std::ofstream tmpStream;
  inputImagePath = IOUtil::CreateTemporaryFile(tmpStream, TEMPLATE_FILENAME, inDir + IOUtil::GetDirectorySeparator());
  tmpStream.close();
  std::size_t found = inputImagePath.find_last_of(IOUtil::GetDirectorySeparator());
  std::string fileName = inputImagePath.substr(found + 1);
  std::string token = fileName.substr(0, fileName.find("_"));
  outDir = IOUtil::CreateTemporaryDirectory("totalseg-out-XXXXXX", this->GetMitkTempDir());
  LabelSetImage::Pointer outputBuffer;
  m_ProgressCommand->SetProgress(20);
  IOUtil::Save(inputAtTimeStep, inputImagePath);
  m_ProgressCommand->SetProgress(50);

  outputImagePath = outDir + IOUtil::GetDirectorySeparator() + token + "_000.nii.gz";
  const bool isSubTask = (this->GetSubTask() != DEFAULT_TOTAL_TASK) && (this->GetSubTask() != DEFAULT_TOTAL_TASK_MRI);
  if (isSubTask)
  {
    outputImagePath = outDir;
    this->run_totalsegmentator(
      spExec, inputImagePath, outputImagePath, !isSubTask, !isSubTask, this->GetGpuId(), this->GetSubTask());
    // Construct Label Id map
    std::vector<std::string> files = SUBTASKS_MAP.at(this->GetSubTask());
    // Agglomerate individual mask files into one multi-label image.
    std::for_each(files.begin(),
                  files.end(),
                  [&](std::string &fileName) { fileName = (outDir + IOUtil::GetDirectorySeparator() + fileName); });
    outputBuffer = AgglomerateLabelFiles(files, inputAtTimeStep->GetDimensions(), inputAtTimeStep->GetGeometry());
  }
  else
  {
    this->run_totalsegmentator(
      spExec, inputImagePath, outputImagePath, this->GetFast(), !isSubTask, this->GetGpuId(), this->GetSubTask());
    Image::Pointer outputImage = IOUtil::Load<Image>(outputImagePath);
    outputBuffer = mitk::LabelSetImage::New();
    outputBuffer->InitializeByLabeledImage(outputImage);
    outputBuffer->SetGeometry(inputAtTimeStep->GetGeometry());
  }
  m_ProgressCommand->SetProgress(180);
  mitk::ImageReadAccessor newMitkImgAcc(outputBuffer.GetPointer());
  this->MapLabelsToSegmentation(outputBuffer, previewImage, m_LabelMapTotal);
  previewImage->SetVolume(newMitkImgAcc.GetData(), timeStep);
}

void mitk::TotalSegmentatorTool::UpdatePrepare()
{
  Superclass::UpdatePrepare();
  auto preview = this->GetPreviewSegmentation();
  preview->RemoveLabels(preview->GetAllLabelValues());
  if (m_LabelMapTotal.empty())
  {
    this->ParseLabelMapTotalDefault();
  }
  const bool isSubTask = (this->GetSubTask() != DEFAULT_TOTAL_TASK) && (this->GetSubTask() != DEFAULT_TOTAL_TASK_MRI);
  if (isSubTask)
  {
    std::vector<std::string> files = SUBTASKS_MAP.at(this->GetSubTask());
    m_LabelMapTotal.clear();
    mitk::Label::PixelType labelId = 1;
    for (auto const &file : files)
    {
      std::string labelName = file.substr(0, file.find('.'));
      m_LabelMapTotal[labelId] = labelName;
      labelId++;
    }
  }
}

mitk::LabelSetImage::Pointer mitk::TotalSegmentatorTool::AgglomerateLabelFiles(std::vector<std::string> &filePaths,
                                                                               const unsigned int *dimensions,
                                                                               mitk::BaseGeometry *geometry)
{
  Label::PixelType labelId = 1;
  auto aggloLabelImage = mitk::LabelSetImage::New();
  auto initImage = mitk::Image::New();
  initImage->Initialize(mitk::MakeScalarPixelType<mitk::Label::PixelType>(), 3, dimensions);
  aggloLabelImage->Initialize(initImage);
  aggloLabelImage->SetGeometry(geometry);
  const auto layerIndex = aggloLabelImage->AddLayer();
  aggloLabelImage->SetActiveLayer(layerIndex);

  for (auto const &outputImagePath : filePaths)
  {
    double rgba[4];
    aggloLabelImage->GetLookupTable()->GetTableValue(labelId, rgba);
    mitk::Color color;
    color.SetRed(rgba[0]);
    color.SetGreen(rgba[1]);
    color.SetBlue(rgba[2]);

    auto label = mitk::Label::New();
    label->SetName("object-" + std::to_string(labelId));
    label->SetValue(labelId);
    label->SetColor(color);
    label->SetOpacity(rgba[3]);

    aggloLabelImage->AddLabel(label, layerIndex, false, false);

    Image::Pointer outputImage = IOUtil::Load<Image>(outputImagePath);
    auto source = mitk::LabelSetImage::New();
    source->InitializeByLabeledImage(outputImage);
    source->SetGeometry(geometry);

    mitk::TransferLabelContent(source, aggloLabelImage, aggloLabelImage->GetConstLabelsByValue(aggloLabelImage->GetLabelValuesByGroup(layerIndex)), 0, 0, false, {{1, labelId}});
    labelId++;
  }
  return aggloLabelImage;
}

void mitk::TotalSegmentatorTool::run_totalsegmentator(ProcessExecutor* spExec,
                                                      const std::string &inputImagePath,
                                                      const std::string &outputImagePath,
                                                      bool isFast,
                                                      bool isMultiLabel,
                                                      unsigned int gpuId,
                                                      const std::string &subTask)
{
  ProcessExecutor::ArgumentListType args;
  std::string command = "TotalSegmentator";
#ifdef _WIN32
  command += ".exe";
#endif
  args.clear();
  args.push_back("-i");
  args.push_back(inputImagePath);

  args.push_back("-o");
  args.push_back(outputImagePath);

  if (subTask == DEFAULT_TOTAL_TASK_MRI)
  {
    args.push_back("--task");
    args.push_back(subTask);
  }
  else if (subTask != DEFAULT_TOTAL_TASK)
  {
    args.push_back("-ta");
    args.push_back(subTask);
  }

  if (isMultiLabel)
  {
    args.push_back("--ml");
  }

  if (isFast)
  {
    args.push_back("--fast");
  }

  try
  {
    std::string cudaEnv = "CUDA_VISIBLE_DEVICES=" + std::to_string(gpuId);
    itksys::SystemTools::PutEnv(cudaEnv.c_str());
    
    std::stringstream logStream;
    for (const auto &arg : args)
      logStream << arg << " ";
    logStream << this->GetPythonPath();
    MITK_INFO << logStream.str();

    spExec->Execute(this->GetPythonPath(), command, args);
  }
  catch (const mitk::Exception &e)
  {
    MITK_ERROR << e.GetDescription();
    return;
  }
}

void mitk::TotalSegmentatorTool::ParseLabelMapTotalDefault()
{
  if (!this->GetLabelMapPath().empty())
  {
    int start_line = 0, end_line = 0;
    if (this->GetSubTask() == DEFAULT_TOTAL_TASK)
      start_line = 111, end_line = 229;
    else if (this->GetSubTask() == DEFAULT_TOTAL_TASK_MRI)
      start_line = 231, end_line = 288;
    std::regex sanitizer(R"([^A-Za-z0-9_])");
    std::fstream newfile;
    newfile.open(this->GetLabelMapPath(), ios::in);
    std::stringstream buffer;
    if (newfile.is_open())
    {
      int line = 0;
      std::string temp;
      while (std::getline(newfile, temp))
      {
        if (line > start_line && line < end_line)
        {
          buffer << temp;
        }
        ++line;
      }
    }
    std::string key, val;
    while (std::getline(std::getline(buffer, key, ':'), val, ','))
    {
      std::string sanitized = std::regex_replace(val, sanitizer, "");
      m_LabelMapTotal[std::stoi(key)] = sanitized;
    }
  }
}

void mitk::TotalSegmentatorTool::MapLabelsToSegmentation(const mitk::LabelSetImage* source,
                                                         mitk::LabelSetImage* dest,
                                                         std::map<mitk::Label::PixelType, std::string> &labelMap)
{
  auto lookupTable = mitk::LookupTable::New();
  lookupTable->SetType(mitk::LookupTable::LookupTableType::MULTILABEL);
  for (auto const &[key, val] : labelMap)
  {
    if (source->ExistLabel(key, source->GetActiveLayer()))
    {
      Label::Pointer label = Label::New(key, val);
      std::array<double, 3> lookupTableColor;
      lookupTable->GetColor(key, lookupTableColor.data());
      Color color;
      color.SetRed(lookupTableColor[0]);
      color.SetGreen(lookupTableColor[1]);
      color.SetBlue(lookupTableColor[2]);
      label->SetColor(color);
      dest->AddLabel(label, 0,false);
    }
  }
}

std::string mitk::TotalSegmentatorTool::GetLabelMapPath()
{
  std::string pythonFileName;
  fs::path pathToLabelMap(this->GetPythonPath());
  pathToLabelMap = pathToLabelMap.parent_path();
#ifdef _WIN32
  pythonFileName = pathToLabelMap.string() + "/Lib/site-packages/totalsegmentator/map_to_binary.py";
#else
  pathToLabelMap.append("lib");
  for (auto const &dir_entry : fs::directory_iterator{pathToLabelMap})
  {
    if (dir_entry.is_directory())
    {
      auto dirName = dir_entry.path().filename().string();
      if (dirName.rfind("python", 0) == 0)
      {
        pathToLabelMap.append(dir_entry.path().filename().string());
        break;
      }
    }
  }
  pythonFileName = pathToLabelMap.string() + "/site-packages/totalsegmentator/map_to_binary.py";
#endif
  return pythonFileName;
}
