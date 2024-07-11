/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

// MITK
#include <mitkIOUtil.h>
#include <mitkInteractionEventConst.h>
#include <mitkInteractionTestHelper.h>
#include <mitkPlaneGeometryDataMapper2D.h>
#include <mitkRenderingManager.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkTimeNavigationController.h>

// VTK
#include <vtkCamera.h>
#include <vtkRenderWindowInteractor.h>

// us
#include <usGetModuleContext.h>

#include <tinyxml2.h>

mitk::InteractionTestHelper::InteractionTestHelper(const std::string &interactionXmlFilePath)
  : m_InteractionFilePath(interactionXmlFilePath)
{
  this->Initialize(interactionXmlFilePath);
}

void mitk::InteractionTestHelper::Initialize(const std::string &interactionXmlFilePath)
{
  tinyxml2::XMLDocument document;
  if (tinyxml2::XML_SUCCESS == document.LoadFile(interactionXmlFilePath.c_str()))
  {
    // create data storage
    m_DataStorage = mitk::StandaloneDataStorage::New();

    // for each renderer found create a render window and configure
    for (auto *element = document.FirstChildElement(mitk::InteractionEventConst::xmlTagInteractions().c_str())
                                   ->FirstChildElement(mitk::InteractionEventConst::xmlTagConfigRoot().c_str())
                                   ->FirstChildElement(mitk::InteractionEventConst::xmlTagRenderer().c_str());
         element != nullptr;
         element = element->NextSiblingElement(mitk::InteractionEventConst::xmlTagRenderer().c_str()))
    {
      // get name of renderer
      const char *rendererName =
        element->Attribute(mitk::InteractionEventConst::xmlEventPropertyRendererName().c_str());

      // get view direction
      mitk::AnatomicalPlane viewDirection = mitk::AnatomicalPlane::Axial;
      if (element->Attribute(mitk::InteractionEventConst::xmlEventPropertyViewDirection().c_str()) != nullptr)
      {
        int viewDirectionNum =
          std::atoi(element->Attribute(mitk::InteractionEventConst::xmlEventPropertyViewDirection().c_str()));
        viewDirection = static_cast<mitk::AnatomicalPlane>(viewDirectionNum);
      }

      // get mapper slot id
      MapperSlotId mapperID = mitk::BaseRenderer::Standard2D;
      if (element->Attribute(mitk::InteractionEventConst::xmlEventPropertyMapperID().c_str()) != nullptr)
      {
        int mapperIDNum =
          std::atoi(element->Attribute(mitk::InteractionEventConst::xmlEventPropertyMapperID().c_str()));
        mapperID = static_cast<MapperSlotId>(mapperIDNum);
      }

      // Get Size of Render Windows
      int size[3];
      size[0] = size[1] = size[2] = 0;
      if (element->Attribute(mitk::InteractionEventConst::xmlRenderSizeX().c_str()) != nullptr)
      {
        size[0] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlRenderSizeX().c_str()));
      }
      if (element->Attribute(mitk::InteractionEventConst::xmlRenderSizeY().c_str()) != nullptr)
      {
        size[1] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlRenderSizeY().c_str()));
      }
      if (element->Attribute(mitk::InteractionEventConst::xmlRenderSizeZ().c_str()) != nullptr)
      {
        size[2] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlRenderSizeZ().c_str()));
      }

      // create renderWindow, renderer and dispatcher
      auto rw = RenderWindow::New(nullptr, rendererName); // VtkRenderWindow is created within constructor if nullptr

      if (size[0] != 0 && size[1] != 0)
      {
        rw->SetSize(size[0], size[1]);
        rw->GetRenderer()->Resize(size[0], size[1]);
      }

      // set storage of renderer
      rw->GetRenderer()->SetDataStorage(m_DataStorage);

      // set view direction to axial
      rw->GetSliceNavigationController()->SetDefaultViewDirection(viewDirection);

      // set renderer to render 2D
      rw->GetRenderer()->SetMapperID(mapperID);

      rw->GetRenderer()->PrepareRender();

      // Some more magic for the 3D render window case:
      // Camera view direction, position and focal point

      if (mapperID == mitk::BaseRenderer::Standard3D)
      {
        if (element->Attribute(mitk::InteractionEventConst::xmlCameraFocalPointX().c_str()) != nullptr)
        {
          double cameraFocalPoint[3];

          cameraFocalPoint[0] =
            std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraFocalPointX().c_str()));
          cameraFocalPoint[1] =
            std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraFocalPointY().c_str()));
          cameraFocalPoint[2] =
            std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraFocalPointZ().c_str()));
          rw->GetRenderer()->GetVtkRenderer()->GetActiveCamera()->SetFocalPoint(cameraFocalPoint);
        }

        if (element->Attribute(mitk::InteractionEventConst::xmlCameraPositionX().c_str()) != nullptr)
        {
          double cameraPosition[3];

          cameraPosition[0] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraPositionX().c_str()));
          cameraPosition[1] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraPositionY().c_str()));
          cameraPosition[2] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraPositionZ().c_str()));
          rw->GetRenderer()->GetVtkRenderer()->GetActiveCamera()->SetPosition(cameraPosition);
        }

        if (element->Attribute(mitk::InteractionEventConst::xmlViewUpX().c_str()) != nullptr)
        {
          double viewUp[3];

          viewUp[0] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlViewUpX().c_str()));
          viewUp[1] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlViewUpY().c_str()));
          viewUp[2] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlViewUpZ().c_str()));
          rw->GetRenderer()->GetVtkRenderer()->GetActiveCamera()->SetViewUp(viewUp);
        }
      }

      rw->GetVtkRenderWindow()->Render();
      rw->GetVtkRenderWindow()->WaitForCompletion();

      // add to list of known render windows
      m_RenderWindowList.push_back(rw);
    }

    // register interaction event obserer to handle scroll events
    InitializeDisplayActionEventHandling();
  }
  else
  {
    mitkThrow() << "Can not load interaction xml file <" << m_InteractionFilePath << ">";
  }

  // WARNING assumes a 3D window exists !!!!
  this->AddDisplayPlaneSubTree();
}

void mitk::InteractionTestHelper::InitializeDisplayActionEventHandling()
{
  m_DisplayActionEventBroadcast = mitk::DisplayActionEventBroadcast::New();
  m_DisplayActionEventBroadcast->LoadStateMachine("DisplayInteraction.xml");
  m_DisplayActionEventBroadcast->SetEventConfig("DisplayConfigMITKBase.xml");
  m_DisplayActionEventBroadcast->AddEventConfig("DisplayConfigCrosshair.xml");
}

mitk::InteractionTestHelper::~InteractionTestHelper()
{
  // unregister renderers
  auto it = m_RenderWindowList.begin();
  auto end = m_RenderWindowList.end();

  for (; it != end; ++it)
  {
    mitk::BaseRenderer::RemoveInstance((*it)->GetVtkRenderWindow());
  }

  mitk::RenderingManager::GetInstance()->RemoveAllObservers();
}

mitk::DataStorage::Pointer mitk::InteractionTestHelper::GetDataStorage()
{
  return m_DataStorage;
}

void mitk::InteractionTestHelper::AddNodeToStorage(mitk::DataNode::Pointer node)
{
  this->m_DataStorage->Add(node);

  this->Set3dCameraSettings();
}

void mitk::InteractionTestHelper::PlaybackInteraction()
{
  mitk::RenderingManager::GetInstance()->InitializeViewsByBoundingObjects(m_DataStorage);
  // load events if not loaded yet
  if (m_Events.empty())
    this->LoadInteraction();

  auto it = m_RenderWindowList.begin();
  auto end = m_RenderWindowList.end();
  for (; it != end; ++it)
  {
    (*it)->GetRenderer()->PrepareRender();

    (*it)->GetVtkRenderWindow()->Render();
    (*it)->GetVtkRenderWindow()->WaitForCompletion();
  }
  mitk::RenderingManager::GetInstance()->InitializeViewsByBoundingObjects(m_DataStorage);

  it = m_RenderWindowList.begin();
  for (; it != end; ++it)
  {
    (*it)->GetVtkRenderWindow()->Render();
    (*it)->GetVtkRenderWindow()->WaitForCompletion();
  }

  // mitk::RenderingManager::GetInstance()->ForceImmediateUpdateAll();
  // playback all events in queue
  for (unsigned long i = 0; i < m_Events.size(); ++i)
  {
    // let dispatcher of sending renderer process the event
    m_Events.at(i)->GetSender()->GetDispatcher()->ProcessEvent(m_Events.at(i));
  }
  if (false)
  {
    it--;
    (*it)->GetVtkRenderWindow()->GetInteractor()->Start();
  }
}

void mitk::InteractionTestHelper::LoadInteraction()
{
  // load interaction pattern from xml file
  std::ifstream xmlStream(m_InteractionFilePath.c_str());
  mitk::XML2EventParser parser(xmlStream);
  m_Events = parser.GetInteractions();
  xmlStream.close();
  // Avoid VTK warning: Trying to delete object with non-zero reference count.
  parser.SetReferenceCount(0);
}

void mitk::InteractionTestHelper::SetTimeStep(int newTimeStep)
{
  auto rm = mitk::RenderingManager::GetInstance();
  rm->InitializeViewsByBoundingObjects(m_DataStorage);

  bool timeStepIsvalid = rm->GetTimeNavigationController()->GetInputWorldTimeGeometry()->IsValidTimeStep(newTimeStep);

  if (timeStepIsvalid)
  {
    rm->GetTimeNavigationController()->GetStepper()->SetPos(newTimeStep);
  }
}

mitk::RenderWindow *mitk::InteractionTestHelper::GetRenderWindowByName(const std::string &name)
{
  auto it = m_RenderWindowList.begin();
  auto end = m_RenderWindowList.end();

  for (; it != end; ++it)
  {
    if (name.compare((*it)->GetRenderer()->GetName()) == 0)
      return (*it).GetPointer();
  }

  return nullptr;
}

mitk::RenderWindow *mitk::InteractionTestHelper::GetRenderWindowByDefaultViewDirection(AnatomicalPlane viewDirection)
{
  auto it = m_RenderWindowList.begin();
  auto end = m_RenderWindowList.end();

  for (; it != end; ++it)
  {
    if (viewDirection == (*it)->GetSliceNavigationController()->GetDefaultViewDirection())
      return (*it).GetPointer();
  }

  return nullptr;
}

mitk::RenderWindow *mitk::InteractionTestHelper::GetRenderWindow(unsigned int index)
{
  if (index < m_RenderWindowList.size())
  {
    return m_RenderWindowList.at(index).GetPointer();
  }
  else
  {
    return nullptr;
  }
}

void mitk::InteractionTestHelper::AddDisplayPlaneSubTree()
{
  // add the displayed planes of the multiwidget to a node to which the subtree
  // @a planesSubTree points ...

  mitk::PlaneGeometryDataMapper2D::Pointer mapper;
  mitk::IntProperty::Pointer layer = mitk::IntProperty::New(1000);

  mitk::DataNode::Pointer node = mitk::DataNode::New();
  node->SetProperty("name", mitk::StringProperty::New("Widgets"));
  node->SetProperty("helper object", mitk::BoolProperty::New(true));

  m_DataStorage->Add(node);

  for (auto it : m_RenderWindowList)
  {
    if (it->GetRenderer()->GetMapperID() == BaseRenderer::Standard3D)
      continue;

    // ... of widget 1
    mitk::DataNode::Pointer planeNode1 =
      (mitk::BaseRenderer::GetInstance(it->GetVtkRenderWindow()))->GetCurrentWorldPlaneGeometryNode();

    planeNode1->SetProperty("visible", mitk::BoolProperty::New(true));
    planeNode1->SetProperty("name", mitk::StringProperty::New("widget1Plane"));
    planeNode1->SetProperty("includeInBoundingBox", mitk::BoolProperty::New(false));
    planeNode1->SetProperty("helper object", mitk::BoolProperty::New(true));
    planeNode1->SetProperty("layer", layer);
    planeNode1->SetColor(1.0, 0.0, 0.0);
    mapper = mitk::PlaneGeometryDataMapper2D::New();
    planeNode1->SetMapper(mitk::BaseRenderer::Standard2D, mapper);
    m_DataStorage->Add(planeNode1, node);
  }
}

void mitk::InteractionTestHelper::Set3dCameraSettings()
{
  tinyxml2::XMLDocument document;
  if (tinyxml2::XML_SUCCESS == document.LoadFile(m_InteractionFilePath.c_str()))
  {
    // for each renderer found create a render window and configure
    for (auto *element = document.FirstChildElement(mitk::InteractionEventConst::xmlTagInteractions().c_str())
                                   ->FirstChildElement(mitk::InteractionEventConst::xmlTagConfigRoot().c_str())
                                   ->FirstChildElement(mitk::InteractionEventConst::xmlTagRenderer().c_str());
         element != nullptr;
         element = element->NextSiblingElement(mitk::InteractionEventConst::xmlTagRenderer().c_str()))
    {
      // get name of renderer
      const char *rendererName =
        element->Attribute(mitk::InteractionEventConst::xmlEventPropertyRendererName().c_str());

      // get mapper slot id
      MapperSlotId mapperID = mitk::BaseRenderer::Standard2D;
      if (element->Attribute(mitk::InteractionEventConst::xmlEventPropertyMapperID().c_str()) != nullptr)
      {
        int mapperIDNum =
          std::atoi(element->Attribute(mitk::InteractionEventConst::xmlEventPropertyMapperID().c_str()));
        mapperID = static_cast<MapperSlotId>(mapperIDNum);
      }

      if (mapperID == mitk::BaseRenderer::Standard3D)
      {
        RenderWindow *namedRenderer = nullptr;

        for (const auto &it : m_RenderWindowList)
        {
          if (strcmp(it->GetRenderer()->GetName(), rendererName) == 0)
          {
            namedRenderer = it.GetPointer();
            break;
          }
        }

        if (namedRenderer == nullptr)
        {
          MITK_ERROR << "No match for render window was found.";
          return;
        }
        namedRenderer->GetRenderer()->PrepareRender();

        if (element->Attribute(mitk::InteractionEventConst::xmlCameraFocalPointX().c_str()) != nullptr)
        {
          double cameraFocalPoint[3];

          cameraFocalPoint[0] =
            std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraFocalPointX().c_str()));
          cameraFocalPoint[1] =
            std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraFocalPointY().c_str()));
          cameraFocalPoint[2] =
            std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraFocalPointZ().c_str()));
          namedRenderer->GetRenderer()->GetVtkRenderer()->GetActiveCamera()->SetFocalPoint(cameraFocalPoint);
        }

        if (element->Attribute(mitk::InteractionEventConst::xmlCameraPositionX().c_str()) != nullptr)
        {
          double cameraPosition[3];

          cameraPosition[0] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraPositionX().c_str()));
          cameraPosition[1] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraPositionY().c_str()));
          cameraPosition[2] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlCameraPositionZ().c_str()));
          namedRenderer->GetRenderer()->GetVtkRenderer()->GetActiveCamera()->SetPosition(cameraPosition);
        }

        if (element->Attribute(mitk::InteractionEventConst::xmlViewUpX().c_str()) != nullptr)
        {
          double viewUp[3];

          viewUp[0] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlViewUpX().c_str()));
          viewUp[1] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlViewUpY().c_str()));
          viewUp[2] = std::atoi(element->Attribute(mitk::InteractionEventConst::xmlViewUpZ().c_str()));
          namedRenderer->GetRenderer()->GetVtkRenderer()->GetActiveCamera()->SetViewUp(viewUp);
        }

        namedRenderer->GetVtkRenderWindow()->Render();
      }
    }
  }
}
