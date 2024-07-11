/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkBaseRenderer.h"
#include "mitkBaseRendererHelper.h"

#include "mitkMapper.h"
#include "mitkResliceMethodProperty.h"

// Geometries
#include "mitkSlicedGeometry3D.h"

#include "mitkVtkLayerController.h"

#include "mitkInteractionConst.h"
#include "mitkProperties.h"
#include "mitkWeakPointerProperty.h"

// VTK
#include <vtkCamera.h>
#include <vtkLinearTransform.h>

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>

namespace mitk
{
  itkEventMacroDefinition(RendererResetEvent, itk::AnyEvent);
}

mitk::BaseRenderer::BaseRendererMapType mitk::BaseRenderer::baseRendererMap;

mitk::BaseRenderer *mitk::BaseRenderer::GetInstance(vtkRenderWindow *renWin)
{
  for (auto mapit = baseRendererMap.begin(); mapit != baseRendererMap.end(); ++mapit)
  {
    if ((*mapit).first == renWin)
      return (*mapit).second;
  }
  return nullptr;
}

void mitk::BaseRenderer::AddInstance(vtkRenderWindow *renWin, BaseRenderer *baseRenderer)
{
  if (renWin == nullptr || baseRenderer == nullptr)
    return;

  // ensure that no BaseRenderer is managed twice
  mitk::BaseRenderer::RemoveInstance(renWin);

  baseRendererMap.insert(BaseRendererMapType::value_type(renWin, baseRenderer));
}

void mitk::BaseRenderer::RemoveInstance(vtkRenderWindow *renWin)
{
  auto mapit = baseRendererMap.find(renWin);
  if (mapit != baseRendererMap.end())
    baseRendererMap.erase(mapit);
}

mitk::BaseRenderer *mitk::BaseRenderer::GetByName(const std::string &name)
{
  for (auto mapit = baseRendererMap.begin(); mapit != baseRendererMap.end(); ++mapit)
  {
    if ((*mapit).second->m_Name == name)
      return (*mapit).second;
  }
  return nullptr;
}

vtkRenderWindow *mitk::BaseRenderer::GetRenderWindowByName(const std::string &name)
{
  for (auto mapit = baseRendererMap.begin(); mapit != baseRendererMap.end(); ++mapit)
  {
    if ((*mapit).second->m_Name == name)
      return (*mapit).first;
  }
  return nullptr;
}

mitk::BaseRenderer::BaseRendererMapType mitk::BaseRenderer::GetSpecificRenderWindows(MapperSlotId mapper)
{
  BaseRendererMapType allRenderWindows;
  for (auto mapit = baseRendererMap.begin(); mapit != baseRendererMap.end(); ++mapit)
  {
    if (mapper == mapit->second->GetMapperID())
    {
      allRenderWindows.insert(BaseRendererMapType::value_type(mapit->first, mapit->second));
    }
  }

  return allRenderWindows;
}

mitk::BaseRenderer::BaseRendererMapType mitk::BaseRenderer::GetAll2DRenderWindows()
{
  return GetSpecificRenderWindows(BaseRenderer::Standard2D);
}

mitk::BaseRenderer::BaseRendererMapType mitk::BaseRenderer::GetAll3DRenderWindows()
{
  return GetSpecificRenderWindows(BaseRenderer::Standard3D);
}

mitk::BaseRenderer::BaseRenderer(const char *name,
                                 vtkRenderWindow *renWin)
  : m_RenderWindow(nullptr),
    m_VtkRenderer(nullptr),
    m_MapperID(StandardMapperSlot::Standard2D),
    m_DataStorage(nullptr),
    m_LastUpdateTime(0),
    m_CameraController(nullptr),
    m_CameraRotationController(nullptr),
    m_SliceNavigationController(nullptr),
    m_WorldTimeGeometry(nullptr),
    m_InteractionReferenceGeometry(nullptr),
    m_CurrentWorldGeometry(nullptr),
    m_CurrentWorldPlaneGeometry(nullptr),
    m_Slice(0),
    m_TimeStep(),
    m_CurrentWorldPlaneGeometryUpdateTime(),
    m_TimeStepUpdateTime(),
    m_KeepDisplayedRegion(true),
    m_ReferenceGeometryAligned(true),
    m_CurrentWorldPlaneGeometryData(nullptr),
    m_CurrentWorldPlaneGeometryNode(nullptr),
    m_CurrentWorldPlaneGeometryTransformTime(0),
    m_Name(name),
    m_EmptyWorldGeometry(true),
    m_NumberOfVisibleLODEnabledMappers(0)
{
  m_Bounds[0] = 0;
  m_Bounds[1] = 0;
  m_Bounds[2] = 0;
  m_Bounds[3] = 0;
  m_Bounds[4] = 0;
  m_Bounds[5] = 0;

  if (name != nullptr)
  {
    m_Name = name;
  }
  else
  {
    m_Name = "unnamed renderer";
    itkWarningMacro(<< "Created unnamed renderer. Bad for serialization. Please choose a name.");
  }

  if (renWin != nullptr)
  {
    m_RenderWindow = renWin;
    m_RenderWindow->Register(nullptr);
  }
  else
  {
    itkWarningMacro(<< "Created mitkBaseRenderer without vtkRenderWindow present.");
  }
  // instances.insert( this );

  // adding this BaseRenderer to the List of all BaseRenderer

  m_BindDispatcherInteractor = new mitk::BindDispatcherInteractor(GetName());

  WeakPointerProperty::Pointer rendererProp = WeakPointerProperty::New((itk::Object *)this);

  m_CurrentWorldPlaneGeometry = mitk::PlaneGeometry::New();

  m_CurrentWorldPlaneGeometryData = mitk::PlaneGeometryData::New();
  m_CurrentWorldPlaneGeometryData->SetPlaneGeometry(m_CurrentWorldPlaneGeometry);
  m_CurrentWorldPlaneGeometryNode = mitk::DataNode::New();
  m_CurrentWorldPlaneGeometryNode->SetData(m_CurrentWorldPlaneGeometryData);
  m_CurrentWorldPlaneGeometryNode->GetPropertyList()->SetProperty("renderer", rendererProp);
  m_CurrentWorldPlaneGeometryNode->GetPropertyList()->SetProperty("layer", IntProperty::New(1000));

  m_CurrentWorldPlaneGeometryNode->SetProperty("reslice.thickslices", mitk::ResliceMethodProperty::New());
  m_CurrentWorldPlaneGeometryNode->SetProperty("reslice.thickslices.num", mitk::IntProperty::New(1));

  m_CurrentWorldPlaneGeometryTransformTime = m_CurrentWorldPlaneGeometryNode->GetVtkTransform()->GetMTime();

  m_SliceNavigationController = mitk::SliceNavigationController::New();
  m_SliceNavigationController->SetRenderer(this);
  m_SliceNavigationController->ConnectGeometrySendEvent(this);
  m_SliceNavigationController->ConnectGeometryUpdateEvent(this);
  m_SliceNavigationController->ConnectGeometrySliceEvent(this);

  auto* timeNavigationController = RenderingManager::GetInstance()->GetTimeNavigationController();
  timeNavigationController->ConnectTimeEvent(this);

  m_CameraRotationController = mitk::CameraRotationController::New();
  m_CameraRotationController->SetRenderWindow(m_RenderWindow);
  m_CameraRotationController->AcquireCamera();

  m_CameraController = mitk::CameraController::New();
  m_CameraController->SetRenderer(this);

  m_VtkRenderer = vtkRenderer::New();
  m_VtkRenderer->SetMaximumNumberOfPeels(16);

  if (AntiAliasing::FastApproximate == RenderingManager::GetInstance()->GetAntiAliasing())
    m_VtkRenderer->UseFXAAOn();

  if (nullptr == mitk::VtkLayerController::GetInstance(m_RenderWindow))
    mitk::VtkLayerController::AddInstance(m_RenderWindow, m_VtkRenderer);

  mitk::VtkLayerController::GetInstance(m_RenderWindow)->InsertSceneRenderer(m_VtkRenderer);
}

mitk::BaseRenderer::~BaseRenderer()
{
  if (m_VtkRenderer != nullptr)
  {
    m_VtkRenderer->Delete();
    m_VtkRenderer = nullptr;
  }

  if (m_CameraController.IsNotNull())
    m_CameraController->SetRenderer(nullptr);

  mitk::VtkLayerController::RemoveInstance(m_RenderWindow);

  RemoveAllLocalStorages();

  m_DataStorage = nullptr;

  if (m_BindDispatcherInteractor != nullptr)
  {
    delete m_BindDispatcherInteractor;
  }

  if (m_RenderWindow != nullptr)
  {
    m_RenderWindow->Delete();
    m_RenderWindow = nullptr;
  }

  auto* timeNavigationController = RenderingManager::GetInstance()->GetTimeNavigationController();
  timeNavigationController->Disconnect(this);
}

void mitk::BaseRenderer::RemoveAllLocalStorages()
{
  this->InvokeEvent(RendererResetEvent());

  std::list<mitk::BaseLocalStorageHandler *>::iterator it;
  for (it = m_RegisteredLocalStorageHandlers.begin(); it != m_RegisteredLocalStorageHandlers.end(); ++it)
    (*it)->ClearLocalStorage(this, false);
  m_RegisteredLocalStorageHandlers.clear();
}

void mitk::BaseRenderer::RegisterLocalStorageHandler(mitk::BaseLocalStorageHandler *lsh)
{
  m_RegisteredLocalStorageHandlers.push_back(lsh);
}

void mitk::BaseRenderer::UnregisterLocalStorageHandler(mitk::BaseLocalStorageHandler *lsh)
{
  m_RegisteredLocalStorageHandlers.remove(lsh);
}

void mitk::BaseRenderer::SetDataStorage(DataStorage *storage)
{
  if (storage != m_DataStorage && storage != nullptr)
  {
    m_DataStorage = storage;
    m_BindDispatcherInteractor->SetDataStorage(m_DataStorage);
    this->Modified();
  }
}

mitk::Dispatcher::Pointer mitk::BaseRenderer::GetDispatcher() const
{
  return m_BindDispatcherInteractor->GetDispatcher();
}

void mitk::BaseRenderer::Resize(int w, int h)
{
  m_RenderWindow->SetSize(w, h);
}

void mitk::BaseRenderer::InitRenderer(vtkRenderWindow *renderwindow)
{
  if (m_RenderWindow != renderwindow)
  {
    if (m_RenderWindow != nullptr)
    {
      m_RenderWindow->Delete();
    }
    m_RenderWindow = renderwindow;
    if (m_RenderWindow != nullptr)
    {
      m_RenderWindow->Register(nullptr);
    }
  }
  RemoveAllLocalStorages();

  if (m_CameraController.IsNotNull())
  {
    m_CameraController->SetRenderer(this);
  }
}

void mitk::BaseRenderer::InitSize(int w, int h)
{
  m_RenderWindow->SetSize(w, h);
}

void mitk::BaseRenderer::SetWorldTimeGeometry(const mitk::TimeGeometry* geometry)
{
  if (m_WorldTimeGeometry == geometry)
  {
    return;
  }

  m_WorldTimeGeometry = geometry;

  this->UpdateCurrentGeometries();
}

void mitk::BaseRenderer::SetInteractionReferenceGeometry(const TimeGeometry* geometry)
{
  if (m_InteractionReferenceGeometry == geometry)
  {
    return;
  }

  m_InteractionReferenceGeometry = geometry;

  this->UpdateCurrentGeometries();
}

void mitk::BaseRenderer::SetSlice(unsigned int slice)
{
  if (m_Slice == slice)
  {
    return;
  }

  m_Slice = slice;

  this->UpdateCurrentGeometries();
}

void mitk::BaseRenderer::SetTimeStep(unsigned int timeStep)
{
  if (m_TimeStep == timeStep)
  {
    return;
  }

  m_TimeStep = timeStep;
  m_TimeStepUpdateTime.Modified();

  this->UpdateCurrentGeometries();
}

mitk::TimeStepType mitk::BaseRenderer::GetTimeStep(const mitk::BaseData* data) const
{
  if ((data == nullptr) || (data->IsInitialized() == false))
  {
    return TIMESTEP_INVALID;
  }
  return data->GetTimeGeometry()->TimePointToTimeStep(GetTime());
}

mitk::ScalarType mitk::BaseRenderer::GetTime() const
{
  if (m_WorldTimeGeometry.IsNull())
  {
    return 0;
  }
  else
  {
    ScalarType timeInMS = m_WorldTimeGeometry->TimeStepToTimePoint(GetTimeStep());
    if (timeInMS == itk::NumericTraits<mitk::ScalarType>::NonpositiveMin())
      return 0;
    else
      return timeInMS;
  }
}

void mitk::BaseRenderer::SetGeometry(const itk::EventObject& geometrySendEvent)
{
  const auto* sendEvent = dynamic_cast<const SliceNavigationController::GeometrySendEvent*>(&geometrySendEvent);

  if (nullptr == sendEvent)
  {
    return;
  }

  SetWorldTimeGeometry(sendEvent->GetTimeGeometry());
}

void mitk::BaseRenderer::UpdateGeometry(const itk::EventObject& geometryUpdateEvent)
{
  const auto* updateEvent = dynamic_cast<const SliceNavigationController::GeometryUpdateEvent*>(&geometryUpdateEvent);

  if (nullptr == updateEvent)
  {
    return;
  }

  if (m_CurrentWorldGeometry.IsNull())
  {
    return;
  }

  const auto* slicedWorldGeometry = dynamic_cast<const SlicedGeometry3D*>(m_CurrentWorldGeometry.GetPointer());
  if (slicedWorldGeometry)
  {
    PlaneGeometry* geometry2D = slicedWorldGeometry->GetPlaneGeometry(m_Slice);

    SetCurrentWorldPlaneGeometry(geometry2D); // calls Modified()
  }
}

void mitk::BaseRenderer::SetGeometrySlice(const itk::EventObject& geometrySliceEvent)
{
  const auto* sliceEvent = dynamic_cast<const SliceNavigationController::GeometrySliceEvent*>(&geometrySliceEvent);

  if (nullptr == sliceEvent)
  {
    return;
  }

  this->SetSlice(sliceEvent->GetPos());
}

void mitk::BaseRenderer::SetGeometryTime(const itk::EventObject& geometryTimeEvent)
{
  const auto* timeEvent = dynamic_cast<const TimeNavigationController::TimeEvent *>(&geometryTimeEvent);

  if (nullptr == timeEvent)
  {
    return;
  }

  this->SetTimeStep(timeEvent->GetTimeStep());
}

void mitk::BaseRenderer::SendUpdateSlice()
{
  m_CurrentWorldPlaneGeometryUpdateTime.Modified();
}

void mitk::BaseRenderer::SetMapperID(MapperSlotId id)
{
  if (m_MapperID != id)
  {
    bool useDepthPeeling = Standard3D == id;
    m_VtkRenderer->SetUseDepthPeeling(useDepthPeeling);
    m_VtkRenderer->SetUseDepthPeelingForVolumes(useDepthPeeling);

    m_MapperID = id;
    this->Modified();
  }
}

int* mitk::BaseRenderer::GetSize() const
{
  return m_RenderWindow->GetSize();
}

int* mitk::BaseRenderer::GetViewportSize() const
{
  return m_VtkRenderer->GetSize();
}

const double* mitk::BaseRenderer::GetBounds() const
{
  return m_Bounds;
}

void mitk::BaseRenderer::RequestUpdate()
{
  SetConstrainZoomingAndPanning(true);
  RenderingManager::GetInstance()->RequestUpdate(m_RenderWindow);
}

void mitk::BaseRenderer::ForceImmediateUpdate()
{
  RenderingManager::GetInstance()->ForceImmediateUpdate(m_RenderWindow);
}

unsigned int mitk::BaseRenderer::GetNumberOfVisibleLODEnabledMappers() const
{
  return m_NumberOfVisibleLODEnabledMappers;
}

void mitk::BaseRenderer::SetSliceNavigationController(mitk::SliceNavigationController *SlicenavigationController)
{
  if (SlicenavigationController == nullptr)
    return;

  // copy worldgeometry
  SlicenavigationController->SetInputWorldTimeGeometry(SlicenavigationController->GetCreatedWorldGeometry());
  SlicenavigationController->Update();

  // set new
  m_SliceNavigationController = SlicenavigationController;
  m_SliceNavigationController->SetRenderer(this);

  if (m_SliceNavigationController.IsNotNull())
  {
    m_SliceNavigationController->ConnectGeometrySendEvent(this);
    m_SliceNavigationController->ConnectGeometryUpdateEvent(this);
    m_SliceNavigationController->ConnectGeometrySliceEvent(this);
  }
}

void mitk::BaseRenderer::DisplayToWorld(const Point2D& displayPoint, Point3D& worldIndex) const
{
  if (m_MapperID == BaseRenderer::Standard2D)
  {
    double display[3], * world;

    // For the right z-position in display coordinates, take the focal point, convert it to display and use it for
    // correct depth.
    double* displayCoord;
    double cameraFP[4];
    // Get camera focal point and position. Convert to display (screen)
    // coordinates. We need a depth value for z-buffer.
    this->GetVtkRenderer()->GetActiveCamera()->GetFocalPoint(cameraFP);
    cameraFP[3] = 0.0;
    this->GetVtkRenderer()->SetWorldPoint(cameraFP[0], cameraFP[1], cameraFP[2], cameraFP[3]);
    this->GetVtkRenderer()->WorldToDisplay();
    displayCoord = this->GetVtkRenderer()->GetDisplayPoint();

    // now convert the display point to world coordinates
    display[0] = displayPoint[0];
    display[1] = displayPoint[1];
    display[2] = displayCoord[2];

    this->GetVtkRenderer()->SetDisplayPoint(display);
    this->GetVtkRenderer()->DisplayToWorld();
    world = this->GetVtkRenderer()->GetWorldPoint();

    for (int i = 0; i < 3; i++)
    {
      worldIndex[i] = world[i] / world[3];
    }
  }
  else if (m_MapperID == BaseRenderer::Standard3D)
  {
    // Seems to be the same code as above, but subclasses may contain different implementations.
    PickWorldPoint(displayPoint, worldIndex);
  }
  return;
}

void mitk::BaseRenderer::DisplayToPlane(const Point2D &displayPoint, Point2D &planePointInMM) const
{
  if (m_MapperID == BaseRenderer::Standard2D)
  {
    Point3D worldPoint;
    this->DisplayToWorld(displayPoint, worldPoint);
    m_CurrentWorldPlaneGeometry->Map(worldPoint, planePointInMM);
  }
  else if (m_MapperID == BaseRenderer::Standard3D)
  {
    MITK_WARN << "No conversion possible with 3D mapper.";
    return;
  }

  return;
}

void mitk::BaseRenderer::WorldToDisplay(const Point3D &worldIndex, Point2D &displayPoint) const
{
  double world[4], *display;

  world[0] = worldIndex[0];
  world[1] = worldIndex[1];
  world[2] = worldIndex[2];
  world[3] = 1.0;

  this->GetVtkRenderer()->SetWorldPoint(world);
  this->GetVtkRenderer()->WorldToDisplay();
  display = this->GetVtkRenderer()->GetDisplayPoint();

  displayPoint[0] = display[0];
  displayPoint[1] = display[1];

  return;
}

void mitk::BaseRenderer::WorldToView(const mitk::Point3D &worldIndex, mitk::Point2D &viewPoint) const
{
  double world[4], *view;

  world[0] = worldIndex[0];
  world[1] = worldIndex[1];
  world[2] = worldIndex[2];
  world[3] = 1.0;

  this->GetVtkRenderer()->SetWorldPoint(world);
  this->GetVtkRenderer()->WorldToView();
  view = this->GetVtkRenderer()->GetViewPoint();
  this->GetVtkRenderer()->ViewToNormalizedViewport(view[0], view[1], view[2]);

  viewPoint[0] = view[0] * this->GetViewportSize()[0];
  viewPoint[1] = view[1] * this->GetViewportSize()[1];

  return;
}

void mitk::BaseRenderer::PlaneToDisplay(const Point2D &planePointInMM, Point2D &displayPoint) const
{
  Point3D worldPoint;
  m_CurrentWorldPlaneGeometry->Map(planePointInMM, worldPoint);
  this->WorldToDisplay(worldPoint, displayPoint);

  return;
}

void mitk::BaseRenderer::PlaneToView(const Point2D &planePointInMM, Point2D &viewPoint) const
{
  Point3D worldPoint;
  m_CurrentWorldPlaneGeometry->Map(planePointInMM, worldPoint);
  this->WorldToView(worldPoint,viewPoint);

  return;
}

double mitk::BaseRenderer::GetScaleFactorMMPerDisplayUnit() const
{
  if (this->GetMapperID() == BaseRenderer::Standard2D)
  {
    // GetParallelScale returns half of the height of the render window in mm.
    // Divided by the half size of the Display size in pixel givest the mm per pixel.
    return this->GetVtkRenderer()->GetActiveCamera()->GetParallelScale() * 2.0 / GetViewportSize()[1];
  }
  else
    return 1.0;
}

mitk::Point2D mitk::BaseRenderer::GetDisplaySizeInMM() const
{
  Point2D dispSizeInMM;
  dispSizeInMM[0] = GetSizeX() * GetScaleFactorMMPerDisplayUnit();
  dispSizeInMM[1] = GetSizeY() * GetScaleFactorMMPerDisplayUnit();
  return dispSizeInMM;
}

mitk::Point2D mitk::BaseRenderer::GetViewportSizeInMM() const
{
  Point2D dispSizeInMM;
  dispSizeInMM[0] = GetViewportSize()[0] * GetScaleFactorMMPerDisplayUnit();
  dispSizeInMM[1] = GetViewportSize()[1] * GetScaleFactorMMPerDisplayUnit();
  return dispSizeInMM;
}

mitk::Point2D mitk::BaseRenderer::GetOriginInMM() const
{
  Point2D originPx;
  originPx[0] = m_VtkRenderer->GetOrigin()[0];
  originPx[1] = m_VtkRenderer->GetOrigin()[1];
  Point2D displayGeometryOriginInMM;
  DisplayToPlane(originPx, displayGeometryOriginInMM); // top left of the render window (Origin)
  return displayGeometryOriginInMM;
}

void mitk::BaseRenderer::SetConstrainZoomingAndPanning(bool constrain)
{
  m_ConstrainZoomingAndPanning = constrain;
  if (m_ConstrainZoomingAndPanning)
  {
    this->GetCameraController()->AdjustCameraToPlane();
  }
}

void mitk::BaseRenderer::UpdateCurrentGeometries()
{
  m_ReferenceGeometryAligned = true;

  if (m_WorldTimeGeometry.IsNull())
  {
    // simply mark the base renderer as modified
    Modified();
    return;
  }

  if (m_TimeStep >= m_WorldTimeGeometry->CountTimeSteps())
  {
    m_TimeStep = m_WorldTimeGeometry->CountTimeSteps() - 1;
  }

  auto slicedWorldGeometry =
    dynamic_cast<SlicedGeometry3D*>(m_WorldTimeGeometry->GetGeometryForTimeStep(m_TimeStep).GetPointer());
  if (slicedWorldGeometry != nullptr)
  {
    if (m_Slice >= slicedWorldGeometry->GetSlices())
    {
      m_Slice = slicedWorldGeometry->GetSlices() - 1;
    }

    SetCurrentWorldGeometry(slicedWorldGeometry);
    SetCurrentWorldPlaneGeometry(slicedWorldGeometry->GetPlaneGeometry(m_Slice));
    m_ReferenceGeometryAligned = BaseRendererHelper::IsRendererGeometryAlignedWithGeometry(this, m_InteractionReferenceGeometry);
  }
}

void mitk::BaseRenderer::SetCurrentWorldPlaneGeometry(const mitk::PlaneGeometry* geometry2d)
{
  if (m_CurrentWorldPlaneGeometry == geometry2d)
  {
    return;
  }

  m_CurrentWorldPlaneGeometry = geometry2d->Clone();
  m_CurrentWorldPlaneGeometryData->SetPlaneGeometry(m_CurrentWorldPlaneGeometry);
  m_CurrentWorldPlaneGeometryUpdateTime.Modified();
  Modified();
}

void mitk::BaseRenderer::SetCurrentWorldGeometry(const mitk::BaseGeometry* geometry)
{
  if (m_CurrentWorldGeometry == geometry)
  {
    return;
  }

  m_CurrentWorldGeometry = geometry;
  if (geometry == nullptr)
  {
    m_Bounds[0] = 0;
    m_Bounds[1] = 0;
    m_Bounds[2] = 0;
    m_Bounds[3] = 0;
    m_Bounds[4] = 0;
    m_Bounds[5] = 0;
    m_EmptyWorldGeometry = true;
    return;
  }

  BoundingBox::Pointer boundingBox = m_CurrentWorldGeometry->CalculateBoundingBoxRelativeToTransform(nullptr);
  const BoundingBox::BoundsArrayType& worldBounds = boundingBox->GetBounds();
  m_Bounds[0] = worldBounds[0];
  m_Bounds[1] = worldBounds[1];
  m_Bounds[2] = worldBounds[2];
  m_Bounds[3] = worldBounds[3];
  m_Bounds[4] = worldBounds[4];
  m_Bounds[5] = worldBounds[5];

  if (boundingBox->GetDiagonalLength2() <= mitk::eps)
  {
    m_EmptyWorldGeometry = true;
  }
  else
  {
    m_EmptyWorldGeometry = false;
  }
}

void mitk::BaseRenderer::PrintSelf(std::ostream &os, itk::Indent indent) const
{
  os << indent << " MapperID: " << m_MapperID << std::endl;
  os << indent << " Slice: " << m_Slice << std::endl;
  os << indent << " TimeStep: " << m_TimeStep << std::endl;

  os << indent << " CurrentWorldPlaneGeometry: ";
  if (m_CurrentWorldPlaneGeometry.IsNull())
    os << "nullptr" << std::endl;
  else
    m_CurrentWorldPlaneGeometry->Print(os, indent);

  os << indent << " CurrentWorldPlaneGeometryUpdateTime: " << m_CurrentWorldPlaneGeometryUpdateTime << std::endl;
  os << indent << " CurrentWorldPlaneGeometryTransformTime: " << m_CurrentWorldPlaneGeometryTransformTime << std::endl;

  Superclass::PrintSelf(os, indent);
}
