/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "QmitkRenderWindow.h"

#include "mitkInteractionKeyEvent.h"
#include "mitkInternalEvent.h"
#include "mitkMouseDoubleClickEvent.h"
#include "mitkMouseMoveEvent.h"
#include "mitkMousePressEvent.h"
#include "mitkMouseReleaseEvent.h"
#include "mitkMouseWheelEvent.h"
#include <mitkStatusBar.h>

#include <QCursor>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QSurfaceFormat>
#include <QTimer>
#include <QWheelEvent>
#include <QWindow>

#include <QmitkMimeTypes.h>
#include <QmitkRenderWindowMenu.h>
#include <QmitkStyleManager.h>

QmitkRenderWindow::QmitkRenderWindow(QWidget *parent, const QString &name, mitk::VtkPropRenderer *)
  : QVTKOpenGLNativeWidget(parent)
  , m_ResendQtEvents(true)
  , m_MenuWidget(nullptr)
  , m_MenuWidgetActivated(false)
  , m_LayoutIndex(QmitkRenderWindowMenu::LayoutIndex::Axial)
  , m_GeometryViolationWarningOverlay(nullptr)
{
  m_InternalRenderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
  m_InternalRenderWindow->SetMultiSamples(0);
  m_InternalRenderWindow->SetAlphaBitPlanes(0);

  setRenderWindow(m_InternalRenderWindow);

  Initialize(name.toStdString().c_str());

  setFocusPolicy(Qt::StrongFocus);
  setMouseTracking(true);
  QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  setSizePolicy(sizePolicy);

  // setup overlay widget to show a warning message with a button
  m_GeometryViolationWarningOverlay = new QmitkButtonOverlayWidget(this);
  m_GeometryViolationWarningOverlay->setVisible(false);
  m_GeometryViolationWarningOverlay->SetOverlayText(
    QStringLiteral("<font color=\"red\"><p style=\"text-align:center\">Interaction is not possible because the "
                   "render window geometry<br>does not match the interaction reference geometry.</p></center></font>"));
  m_GeometryViolationWarningOverlay->SetButtonText("Reset geometry");
  m_GeometryViolationWarningOverlay->SetButtonIcon(QmitkStyleManager::ThemeIcon(QLatin1String(":/Qmitk/reset.svg")));

  connect(m_GeometryViolationWarningOverlay, &QmitkButtonOverlayWidget::Clicked,
          this, &QmitkRenderWindow::ResetGeometry);
}

QmitkRenderWindow::~QmitkRenderWindow()
{
  Destroy(); // Destroy mitkRenderWindowBase
}

void QmitkRenderWindow::SetResendQtEvents(bool resend)
{
  m_ResendQtEvents = resend;
}

void QmitkRenderWindow::SetLayoutIndex(QmitkRenderWindowMenu::LayoutIndex layoutIndex)
{
  m_LayoutIndex = layoutIndex;
  if (nullptr != m_MenuWidget)
  {
    m_MenuWidget->SetLayoutIndex(layoutIndex);
  }
}

QmitkRenderWindowMenu::LayoutIndex QmitkRenderWindow::GetLayoutIndex()
{
  if (nullptr != m_MenuWidget)
  {
    return m_MenuWidget->GetLayoutIndex();
  }
  else
  {
    return QmitkRenderWindowMenu::LayoutIndex::Axial;
  }
}

void QmitkRenderWindow::UpdateLayoutDesignList(QmitkRenderWindowMenu::LayoutDesign layoutDesign)
{
  if (nullptr != m_MenuWidget)
  {
    m_MenuWidget->UpdateLayoutDesignList(layoutDesign);
  }
}

void QmitkRenderWindow::UpdateCrosshairVisibility(bool visible)
{
  m_MenuWidget->UpdateCrosshairVisibility(visible);
}

void QmitkRenderWindow::UpdateCrosshairRotationMode(int mode)
{
  m_MenuWidget->UpdateCrosshairRotationMode(mode);
}

void QmitkRenderWindow::ActivateMenuWidget(bool state)
{
  if (nullptr == m_MenuWidget)
  {
    m_MenuWidget = new QmitkRenderWindowMenu(this, {}, m_Renderer);
    m_MenuWidget->SetLayoutIndex(m_LayoutIndex);
  }

  if (m_MenuWidgetActivated == state)
  {
    // no new state; nothing to do
    return;
  }

  m_MenuWidgetActivated = state;

  if (m_MenuWidgetActivated)
  {
    connect(m_MenuWidget, &QmitkRenderWindowMenu::LayoutDesignChanged, this, &QmitkRenderWindow::LayoutDesignChanged);
    connect(m_MenuWidget, &QmitkRenderWindowMenu::ResetView, this, &QmitkRenderWindow::ResetView);
    connect(m_MenuWidget, &QmitkRenderWindowMenu::CrosshairVisibilityChanged, this, &QmitkRenderWindow::CrosshairVisibilityChanged);
    connect(m_MenuWidget, &QmitkRenderWindowMenu::CrosshairRotationModeChanged, this, &QmitkRenderWindow::CrosshairRotationModeChanged);
  }
  else
  {
    disconnect(m_MenuWidget, &QmitkRenderWindowMenu::LayoutDesignChanged, this, &QmitkRenderWindow::LayoutDesignChanged);
    disconnect(m_MenuWidget, &QmitkRenderWindowMenu::ResetView, this, &QmitkRenderWindow::ResetView);
    disconnect(m_MenuWidget, &QmitkRenderWindowMenu::CrosshairVisibilityChanged, this, &QmitkRenderWindow::CrosshairVisibilityChanged);
    disconnect(m_MenuWidget, &QmitkRenderWindowMenu::CrosshairRotationModeChanged, this, &QmitkRenderWindow::CrosshairRotationModeChanged);

    m_MenuWidget->hide();
  }
}

void QmitkRenderWindow::ShowOverlayMessage(bool show)
{
  m_GeometryViolationWarningOverlay->setVisible(show);
}

void QmitkRenderWindow::moveEvent(QMoveEvent *event)
{
  QVTKOpenGLNativeWidget::moveEvent(event);

  // after a move the overlays need to be positioned
  emit moved();
}

void QmitkRenderWindow::showEvent(QShowEvent *event)
{
  QVTKOpenGLNativeWidget::showEvent(event);

  // this singleshot is necessary to have the overlays positioned correctly after initial show
  // simple call of moved() is no use here!!
  QTimer::singleShot(0, this, SIGNAL(moved()));
}

bool QmitkRenderWindow::event(QEvent* e)
{
  mitk::InteractionEvent::Pointer mitkEvent = nullptr;
  mitk::Point2D mousePosition;
  bool updateStatusBar = false;
  switch (e->type())
  {
    case QEvent::TouchBegin:
      // Starting with Qt 6, using the touchpad on a MacBook for example results in mouse move events with the
      // left mouse button pressed. Hence, hovering over the 3-d render window will already result in rotating the
      // scene. In theory, this can be prevented by either disabling the WA_AcceptTouchEvents attribute for this widget
      // or by globally disabling the AA_SynthesizeMouseForUnhandledTouchEvents attribute for the whole application.
      // Yet, here we are as a last resort, acknowleding touch events just to reject them.
      return false;

    case QEvent::TouchCancel:
      return true;

    case QEvent::MouseMove:
    {
      auto me = static_cast<QMouseEvent *>(e);
      mousePosition = this->GetMousePosition(me);
      mitkEvent = mitk::MouseMoveEvent::New(m_Renderer, mousePosition, GetButtonState(me), GetModifiers(me));
      updateStatusBar = true;
      break;
    }
    case QEvent::MouseButtonPress:
    {
      auto me = static_cast<QMouseEvent *>(e);
      mitkEvent = mitk::MousePressEvent::New( m_Renderer, GetMousePosition(me), GetButtonState(me), GetModifiers(me), GetEventButton(me));
      break;
    }
    case QEvent::MouseButtonRelease:
    {
      auto me = static_cast<QMouseEvent *>(e);
      mitkEvent = mitk::MouseReleaseEvent::New( m_Renderer, GetMousePosition(me), GetButtonState(me), GetModifiers(me), GetEventButton(me));
      break;
    }
    case QEvent::MouseButtonDblClick:
    {
      auto me = static_cast<QMouseEvent *>(e);
      mitkEvent = mitk::MouseDoubleClickEvent::New( m_Renderer, GetMousePosition(me), GetButtonState(me), GetModifiers(me), GetEventButton(me));
      break;
    }
    case QEvent::Wheel:
    {
      auto we = static_cast<QWheelEvent *>(e);
      mousePosition = this->GetMousePosition(we);
      mitkEvent = mitk::MouseWheelEvent::New(m_Renderer, mousePosition, GetButtonState(we), GetModifiers(we), GetDelta(we));
      updateStatusBar = true;
      break;
    }
    case QEvent::KeyPress:
    {
      auto ke = static_cast<QKeyEvent*>(e);
      mitkEvent = mitk::InteractionKeyEvent::New(m_Renderer, GetKeyLetter(ke), GetModifiers(ke));
      break;
    }
    case QEvent::Resize:
    {
      if (nullptr != m_MenuWidget)
        m_MenuWidget->MoveWidgetToCorrectPos();
    }
    default:
    {
      break;
    }
  }

  if (mitkEvent != nullptr)
  {
    if (this->HandleEvent(mitkEvent.GetPointer())) {
      return m_ResendQtEvents ? false : true;
    }
  }

  if (updateStatusBar)
  {
    this->UpdateStatusBar(mousePosition);
  }

  return QVTKOpenGLNativeWidget::event(e);
}

void QmitkRenderWindow::enterEvent(QEnterEvent *e)
{
  auto* baseRenderer = mitk::BaseRenderer::GetInstance(this->GetVtkRenderWindow());
  this->ShowOverlayMessage(!baseRenderer->GetReferenceGeometryAligned());

  if (nullptr != m_MenuWidget)
    m_MenuWidget->ShowMenu();

  QVTKOpenGLNativeWidget::enterEvent(e);
}

void QmitkRenderWindow::leaveEvent(QEvent *e)
{
  auto statusBar = mitk::StatusBar::GetInstance();
  statusBar->DisplayGreyValueText("");
  this->ShowOverlayMessage(false);

  if (nullptr != m_MenuWidget)
    m_MenuWidget->HideMenu();

  QVTKOpenGLNativeWidget::leaveEvent(e);
}

void QmitkRenderWindow::resizeGL(int w, int h)
{
  QVTKOpenGLNativeWidget::resizeGL(w, h);
  mitk::RenderingManager::GetInstance()->ForceImmediateUpdate(renderWindow());
}

void QmitkRenderWindow::dragEnterEvent(QDragEnterEvent *event)
{
  if (event->mimeData()->hasFormat("application/x-mitk-datanodes"))
  {
    event->accept();
  }
}

void QmitkRenderWindow::dropEvent(QDropEvent *event)
{
  QList<mitk::DataNode *> dataNodeList = QmitkMimeTypes::ToDataNodePtrList(event->mimeData());
  if (!dataNodeList.empty())
  {
    std::vector dataNodes(dataNodeList.begin(), dataNodeList.end());
    emit NodesDropped(this, dataNodes);
  }
}

void QmitkRenderWindow::DeferredHideMenu()
{
  MITK_DEBUG << "QmitkRenderWindow::DeferredHideMenu";

  if (nullptr != m_MenuWidget)
  {
    m_MenuWidget->HideMenu();
  }
}

mitk::Point2D QmitkRenderWindow::GetMousePosition(QMouseEvent *me) const
{
  mitk::Point2D point;
  const auto scale = this->devicePixelRatioF();
  const auto position = me->position();
  point[0] = position.x()*scale;
  // We need to convert the y component, as the display and vtk have other definitions for the y direction
  point[1] = m_Renderer->GetSizeY() - position.y()*scale;
  return point;
}

mitk::Point2D QmitkRenderWindow::GetMousePosition(QWheelEvent *we) const
{
  mitk::Point2D point;
  const auto scale = this->devicePixelRatioF();
  const auto position = we->position();
  point[0] = position.x()*scale;
  // We need to convert the y component, as the display and vtk have other definitions for the y direction
  point[1] = m_Renderer->GetSizeY() - position.y()*scale;
  return point;
}

mitk::InteractionEvent::MouseButtons QmitkRenderWindow::GetEventButton(QMouseEvent *me) const
{
  mitk::InteractionEvent::MouseButtons eventButton;
  switch (me->button())
  {
  case Qt::LeftButton:
    eventButton = mitk::InteractionEvent::LeftMouseButton;
    break;
  case Qt::RightButton:
    eventButton = mitk::InteractionEvent::RightMouseButton;
    break;
  case Qt::MiddleButton:
    eventButton = mitk::InteractionEvent::MiddleMouseButton;
    break;
  default:
    eventButton = mitk::InteractionEvent::NoButton;
    break;
  }
  return eventButton;
}

mitk::InteractionEvent::MouseButtons QmitkRenderWindow::GetButtonState(QMouseEvent *me) const
{
  mitk::InteractionEvent::MouseButtons buttonState = mitk::InteractionEvent::NoButton;

  if (me->buttons() & Qt::LeftButton)
  {
    buttonState = buttonState | mitk::InteractionEvent::LeftMouseButton;
  }
  if (me->buttons() & Qt::RightButton)
  {
    buttonState = buttonState | mitk::InteractionEvent::RightMouseButton;
  }
  if (me->buttons() & Qt::MiddleButton)
  {
    buttonState = buttonState | mitk::InteractionEvent::MiddleMouseButton;
  }
  return buttonState;
}

mitk::InteractionEvent::ModifierKeys QmitkRenderWindow::GetModifiers(QInputEvent *me) const
{
  mitk::InteractionEvent::ModifierKeys modifiers = mitk::InteractionEvent::NoKey;

  if (me->modifiers() & Qt::ALT)
  {
    modifiers = modifiers | mitk::InteractionEvent::AltKey;
  }
  if (me->modifiers() & Qt::CTRL)
  {
    modifiers = modifiers | mitk::InteractionEvent::ControlKey;
  }
  if (me->modifiers() & Qt::SHIFT)
  {
    modifiers = modifiers | mitk::InteractionEvent::ShiftKey;
  }
  return modifiers;
}

mitk::InteractionEvent::MouseButtons QmitkRenderWindow::GetButtonState(QWheelEvent *we) const
{
  mitk::InteractionEvent::MouseButtons buttonState = mitk::InteractionEvent::NoButton;

  if (we->buttons() & Qt::LeftButton)
  {
    buttonState = buttonState | mitk::InteractionEvent::LeftMouseButton;
  }
  if (we->buttons() & Qt::RightButton)
  {
    buttonState = buttonState | mitk::InteractionEvent::RightMouseButton;
  }
  if (we->buttons() & Qt::MiddleButton)
  {
    buttonState = buttonState | mitk::InteractionEvent::MiddleMouseButton;
  }
  return buttonState;
}

std::string QmitkRenderWindow::GetKeyLetter(QKeyEvent *ke) const
{
  // Converting Qt Key Event to string element.
  std::string key = "";
  int tkey = ke->key();
  if (tkey < 128)
  { // standard ascii letter
    key = (char)toupper(tkey);
  }
  else
  { // special keys
    switch (tkey)
    {
    case Qt::Key_Return:
      key = mitk::InteractionEvent::KeyReturn;
      break;
    case Qt::Key_Enter:
      key = mitk::InteractionEvent::KeyEnter;
      break;
    case Qt::Key_Escape:
      key = mitk::InteractionEvent::KeyEsc;
      break;
    case Qt::Key_Delete:
      key = mitk::InteractionEvent::KeyDelete;
      break;
    case Qt::Key_Up:
      key = mitk::InteractionEvent::KeyArrowUp;
      break;
    case Qt::Key_Down:
      key = mitk::InteractionEvent::KeyArrowDown;
      break;
    case Qt::Key_Left:
      key = mitk::InteractionEvent::KeyArrowLeft;
      break;
    case Qt::Key_Right:
      key = mitk::InteractionEvent::KeyArrowRight;
      break;

    case Qt::Key_F1:
      key = mitk::InteractionEvent::KeyF1;
      break;
    case Qt::Key_F2:
      key = mitk::InteractionEvent::KeyF2;
      break;
    case Qt::Key_F3:
      key = mitk::InteractionEvent::KeyF3;
      break;
    case Qt::Key_F4:
      key = mitk::InteractionEvent::KeyF4;
      break;
    case Qt::Key_F5:
      key = mitk::InteractionEvent::KeyF5;
      break;
    case Qt::Key_F6:
      key = mitk::InteractionEvent::KeyF6;
      break;
    case Qt::Key_F7:
      key = mitk::InteractionEvent::KeyF7;
      break;
    case Qt::Key_F8:
      key = mitk::InteractionEvent::KeyF8;
      break;
    case Qt::Key_F9:
      key = mitk::InteractionEvent::KeyF9;
      break;
    case Qt::Key_F10:
      key = mitk::InteractionEvent::KeyF10;
      break;
    case Qt::Key_F11:
      key = mitk::InteractionEvent::KeyF11;
      break;
    case Qt::Key_F12:
      key = mitk::InteractionEvent::KeyF12;
      break;

    case Qt::Key_End:
      key = mitk::InteractionEvent::KeyEnd;
      break;
    case Qt::Key_Home:
      key = mitk::InteractionEvent::KeyPos1;
      break;
    case Qt::Key_Insert:
      key = mitk::InteractionEvent::KeyInsert;
      break;
    case Qt::Key_PageDown:
      key = mitk::InteractionEvent::KeyPageDown;
      break;
    case Qt::Key_PageUp:
      key = mitk::InteractionEvent::KeyPageUp;
      break;
    case Qt::Key_Space:
      key = mitk::InteractionEvent::KeySpace;
      break;
    }
  }
  return key;
}

int QmitkRenderWindow::GetDelta(QWheelEvent *we) const
{
  return we->angleDelta().y();
}

void QmitkRenderWindow::UpdateStatusBar(mitk::Point2D pointerPositionOnScreen)
{
  mitk::Point3D worldPosition;
  m_Renderer->ForceImmediateUpdate();
  m_Renderer->DisplayToWorld(pointerPositionOnScreen, worldPosition);
  auto statusBar = mitk::StatusBar::GetInstance();
  statusBar->DisplayRendererInfo(worldPosition, m_Renderer->GetTime());
}
