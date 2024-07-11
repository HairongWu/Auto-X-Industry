/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkPluginActivator.h"
#include "DicomEventHandler.h"
#include <service/event/ctkEventConstants.h>
#include <ctkDictionary.h>
#include <mitkLog.h>
#include <mitkDataNode.h>
#include <mitkIDataStorageService.h>
#include <service/event/ctkEventAdmin.h>
#include <ctkServiceReference.h>
#include <mitkRenderingManager.h>
#include <QVector>
#include "mitkImage.h"
#include <mitkContourModelSet.h>
#include <mitkFileReaderRegistry.h>
#include <mitkDICOMRTMimeTypes.h>

#include <mitkDICOMFileReaderSelector.h>
#include <mitkDICOMDCMTKTagScanner.h>
#include <mitkDICOMEnums.h>
#include <mitkDICOMTagsOfInterestHelper.h>
#include <mitkDICOMProperty.h>
#include <mitkPropertyNameHelper.h>
#include "mitkBaseDICOMReaderService.h"

#include <mitkRTConstants.h>
#include <mitkIsoDoseLevelCollections.h>
#include <mitkIsoDoseLevelSetProperty.h>
#include <mitkIsoDoseLevelVectorProperty.h>
#include <mitkDoseImageVtkMapper2D.h>
#include <mitkRTUIConstants.h>
#include <mitkIsoLevelsGenerator.h>
#include <mitkDoseNodeHelper.h>

#include <vtkSmartPointer.h>
#include <vtkMath.h>
#include <mitkTransferFunction.h>
#include <mitkTransferFunctionProperty.h>
#include <mitkRenderingModeProperty.h>
#include <mitkLocaleSwitch.h>
#include <mitkIOUtil.h>

#include <mitkCoreServices.h>
#include <mitkIPreferencesService.h>
#include <mitkIPreferences.h>

#include <ImporterUtil.h>

namespace
{
  mitk::IFileReader* GetReader(mitk::FileReaderRegistry& readerRegistry, const mitk::CustomMimeType& mimeType)
  {
    try
    {
      return readerRegistry.GetReaders(mitk::MimeType(mimeType, -1, -1)).at(0);
    }
    catch (const std::out_of_range&)
    {
      mitkThrow() << "Cannot find " << mimeType.GetCategory() << " " << mimeType.GetComment() << " file reader.";
    }
  }
}

DicomEventHandler::DicomEventHandler()
{
}

DicomEventHandler::~DicomEventHandler()
{
}

void DicomEventHandler::OnSignalAddSeriesToDataManager(const ctkEvent& ctkEvent)
{
  QStringList listOfFilesForSeries;
  listOfFilesForSeries = ctkEvent.getProperty("FilesForSeries").toStringList();

  if (!listOfFilesForSeries.isEmpty())
  {
    //for rt data, if the modality tag isn't defined or is "CT" the image is handled like before
    if(ctkEvent.containsProperty("Modality") &&
       (ctkEvent.getProperty("Modality").toString().compare("RTDOSE",Qt::CaseInsensitive) == 0 ||
        ctkEvent.getProperty("Modality").toString().compare("RTSTRUCT",Qt::CaseInsensitive) == 0 ||
        ctkEvent.getProperty("Modality").toString().compare("RTPLAN", Qt::CaseInsensitive) == 0))
    {
      QString modality = ctkEvent.getProperty("Modality").toString();
      mitk::FileReaderRegistry readerRegistry;

      if(modality.compare("RTDOSE",Qt::CaseInsensitive) == 0)
      {
          auto doseReader = GetReader(readerRegistry, mitk::DICOMRTMimeTypes::DICOMRT_DOSE_MIMETYPE());
          doseReader->SetInput(ImporterUtil::getUTF8String(listOfFilesForSeries.front()));
          std::vector<itk::SmartPointer<mitk::BaseData> > readerOutput = doseReader->Read();
          if (!readerOutput.empty()){
            mitk::Image::Pointer doseImage = dynamic_cast<mitk::Image*>(readerOutput.at(0).GetPointer());

            mitk::DataNode::Pointer doseImageNode = mitk::DataNode::New();
            doseImageNode->SetData(doseImage);
            doseImageNode->SetName("RTDose");

            if (doseImage != nullptr)
            {
                std::string sopUID;
                if (mitk::GetBackwardsCompatibleDICOMProperty(0x0008, 0x0016, "dicomseriesreader.SOPClassUID", doseImage->GetPropertyList(), sopUID))
                {
                  doseImageNode->SetName(sopUID);
                };

                auto* prefService = mitk::CoreServices::GetPreferencesService();
                auto* prefNode = prefService->GetSystemPreferences()->Node(mitk::RTUIConstants::ROOT_DOSE_VIS_PREFERENCE_NODE_ID);

                if (prefNode == nullptr)
                    mitkThrow() << "Error in preference interface. Cannot find preset node under given name. Name: " << prefNode;

                //set some specific colorwash and isoline properties
                bool showColorWashGlobal = prefNode->GetBool(mitk::RTUIConstants::GLOBAL_VISIBILITY_COLORWASH_ID, true);

                //Set reference dose property
                double referenceDose = prefNode->GetDouble(mitk::RTUIConstants::REFERENCE_DOSE_ID, mitk::RTUIConstants::DEFAULT_REFERENCE_DOSE_VALUE);

                mitk::ConfigureNodeAsDoseNode(doseImageNode, mitk::GenerateIsoLevels_Virtuos(), referenceDose, showColorWashGlobal);

                ctkServiceReference serviceReference = mitk::PluginActivator::getContext()->getServiceReference<mitk::IDataStorageService>();
                mitk::IDataStorageService* storageService = mitk::PluginActivator::getContext()->getService<mitk::IDataStorageService>(serviceReference);
                mitk::DataStorage* dataStorage = storageService->GetDefaultDataStorage().GetPointer()->GetDataStorage();

                dataStorage->Add(doseImageNode);

                mitk::RenderingManager::GetInstance()->InitializeViewsByBoundingObjects(dataStorage);
            }
        }//END DOSE
      }
      else if(modality.compare("RTSTRUCT",Qt::CaseInsensitive) == 0)
      {
          auto structReader = GetReader(readerRegistry, mitk::DICOMRTMimeTypes::DICOMRT_STRUCT_MIMETYPE());
          structReader->SetInput(ImporterUtil::getUTF8String(listOfFilesForSeries.front()));
          std::vector<itk::SmartPointer<mitk::BaseData> > readerOutput = structReader->Read();

          if (readerOutput.empty()){
              MITK_ERROR << "No structure sets were created" << endl;
          }
          else {
              std::vector<mitk::DataNode::Pointer> modelVector;

              ctkServiceReference serviceReference = mitk::PluginActivator::getContext()->getServiceReference<mitk::IDataStorageService>();
              mitk::IDataStorageService* storageService = mitk::PluginActivator::getContext()->getService<mitk::IDataStorageService>(serviceReference);
              mitk::DataStorage* dataStorage = storageService->GetDefaultDataStorage().GetPointer()->GetDataStorage();

              for (const auto& aStruct : readerOutput){
                  mitk::ContourModelSet::Pointer countourModelSet = dynamic_cast<mitk::ContourModelSet*>(aStruct.GetPointer());

                  mitk::DataNode::Pointer structNode = mitk::DataNode::New();
                  structNode->SetData(countourModelSet);
                  structNode->SetProperty("name", aStruct->GetProperty("name"));
                  structNode->SetProperty("color", aStruct->GetProperty("contour.color"));
                  structNode->SetProperty("contour.color", aStruct->GetProperty("contour.color"));
                  structNode->SetProperty("includeInBoundingBox", mitk::BoolProperty::New(false));

                  dataStorage->Add(structNode);
              }
              mitk::RenderingManager::GetInstance()->InitializeViewsByBoundingObjects(dataStorage);
          }
      }
      else if (modality.compare("RTPLAN", Qt::CaseInsensitive) == 0)
      {
          auto planReader = GetReader(readerRegistry, mitk::DICOMRTMimeTypes::DICOMRT_PLAN_MIMETYPE());
          planReader->SetInput(ImporterUtil::getUTF8String(listOfFilesForSeries.front()));
          std::vector<itk::SmartPointer<mitk::BaseData> > readerOutput = planReader->Read();
          if (!readerOutput.empty()){
              //there is no image, only the properties are interesting
              mitk::Image::Pointer planDummyImage = dynamic_cast<mitk::Image*>(readerOutput.at(0).GetPointer());

              mitk::DataNode::Pointer planImageNode = mitk::DataNode::New();
              planImageNode->SetData(planDummyImage);
              planImageNode->SetName("RTPlan");

              ctkServiceReference serviceReference = mitk::PluginActivator::getContext()->getServiceReference<mitk::IDataStorageService>();
              mitk::IDataStorageService* storageService = mitk::PluginActivator::getContext()->getService<mitk::IDataStorageService>(serviceReference);
              mitk::DataStorage* dataStorage = storageService->GetDefaultDataStorage().GetPointer()->GetDataStorage();

              dataStorage->Add(planImageNode);
          }
      }
    }
    else
    {
      mitk::StringList seriesToLoad;
      QStringListIterator it(listOfFilesForSeries);

      while (it.hasNext())
      {
		  seriesToLoad.push_back(ImporterUtil::getUTF8String(it.next()));
      }

      //Get Reference for default data storage.
      ctkServiceReference serviceReference = mitk::PluginActivator::getContext()->getServiceReference<mitk::IDataStorageService>();
      mitk::IDataStorageService* storageService = mitk::PluginActivator::getContext()->getService<mitk::IDataStorageService>(serviceReference);
      mitk::DataStorage* dataStorage = storageService->GetDefaultDataStorage().GetPointer()->GetDataStorage();

      std::vector<mitk::BaseData::Pointer> baseDatas = mitk::IOUtil::Load(seriesToLoad.front());
      for (const auto &data : baseDatas)
      {
        mitk::DataNode::Pointer node = mitk::DataNode::New();
        node->SetData(data);

        std::string nodeName = mitk::DataNode::NO_NAME_VALUE();

        auto nameDataProp = data->GetProperty("name");
        if (nameDataProp.IsNotNull())
        { //if data has a name property set by reader, use this name
          nodeName = nameDataProp->GetValueAsString();
        }
        else
        { //reader didn't specify a name, generate one.
          nodeName = mitk::GenerateNameFromDICOMProperties(node);
        }

        node->SetName(nodeName);

        dataStorage->Add(node);
      }
    }
  }
  else
  {
    MITK_INFO << "There are no files for the current series";
  }
}

void DicomEventHandler::OnSignalRemoveSeriesFromStorage(const ctkEvent& /*ctkEvent*/)
{
}

void DicomEventHandler::SubscribeSlots()
{
  ctkServiceReference ref = mitk::PluginActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::PluginActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "org/mitk/gui/qt/dicom/ADD";
    eventAdmin->subscribeSlot(this, SLOT(OnSignalAddSeriesToDataManager(ctkEvent)), properties);
    properties[ctkEventConstants::EVENT_TOPIC] = "org/mitk/gui/qt/dicom/DELETED";
    eventAdmin->subscribeSlot(this, SLOT(OnSignalRemoveSeriesFromStorage(ctkEvent)), properties);
  }
}
