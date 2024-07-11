/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "QmitkMultiWidgetDecorationManager.h"

#include <mitkIPreferences.h>

// org mitk gui common plugin
#include <mitkIRenderWindowPart.h>

// mitk annotation module
#include <mitkManualPlacementAnnotationRenderer.h>

// mitk qt widgets module
#include <QmitkRenderWindowWidget.h>

// vtk
#include <vtkQImageToImageSource.h>

// qt
#include <QColor>

QmitkMultiWidgetDecorationManager::QmitkMultiWidgetDecorationManager(QmitkAbstractMultiWidget* multiWidget)
  : m_MultiWidget(multiWidget)
  , m_LogoAnnotation(mitk::LogoAnnotation::New())
{
  // nothing here
}

void QmitkMultiWidgetDecorationManager::DecorationPreferencesChanged(const mitk::IPreferences* preferences)
{
  // Enable change of logo. If no DepartmentLogo was set explicitly, MBILogo is used.
  // Set new department logo by prefs->Set("DepartmentLogo", "PathToImage");

  // If no logo was set for this plug-in specifically, walk the parent preference nodes
  // and lookup a logo value there.

  // Disable the logo first, otherwise setting a new logo will have no effect due to how mitkManufacturerLogo works
  ShowLogo(false);
  SetupLogo(qPrintable(":/org.mitk.gui.qt.stdmultiwidgeteditor/defaultWatermark.png"));
  ShowLogo(true);

  const auto* currentNode = preferences;
  while (currentNode)
  {
    bool logoFound = false;
    for(const auto& key : currentNode->Keys())
    {
      if (key == "DepartmentLogo")
      {
        ShowLogo(false);
        auto departmentLogoLocation = currentNode->Get("DepartmentLogo", "");
        if (!departmentLogoLocation.empty())
        {
          SetupLogo(departmentLogoLocation.c_str());
          ShowLogo(true);
        }
        logoFound = true;
        break;
      }
    }

    if (logoFound)
    {
      break;
    }
    currentNode = currentNode->Parent();
  }

  /*
  QmitkMultiWidgetDecorationManager::Colormap colormap = static_cast<QmitkMultiWidgetDecorationManager::Colormap>(preferences->GetInt("Render window widget colormap", 0));
  SetColormap(colormap);
  */

  // show colored rectangle
  ShowAllColoredRectangles(true);

  // show all gradient background
  ShowAllGradientBackgrounds(true);

  // show corner annotations
  ShowAllCornerAnnotations(true);
}

void QmitkMultiWidgetDecorationManager::ShowDecorations(bool show, const QStringList& decorations)
{
  if (nullptr == m_MultiWidget)
  {
    return;
  }

  if (decorations.isEmpty() || decorations.contains(mitk::IRenderWindowPart::DECORATION_BORDER))
  {
    ShowAllColoredRectangles(show);
  }
  if (decorations.isEmpty() || decorations.contains(mitk::IRenderWindowPart::DECORATION_LOGO))
  {
    ShowLogo(show);
  }
  if (decorations.isEmpty() || decorations.contains(mitk::IRenderWindowPart::DECORATION_MENU))
  {
    //m_MultiWidget->ActivateAllRenderWindowMenus(show);
  }
  if (decorations.isEmpty() || decorations.contains(mitk::IRenderWindowPart::DECORATION_BACKGROUND))
  {
    ShowAllGradientBackgrounds(show);
  }
  if (decorations.isEmpty() || decorations.contains(mitk::IRenderWindowPart::DECORATION_CORNER_ANNOTATION))
  {
    ShowAllCornerAnnotations(show);
  }
}

bool QmitkMultiWidgetDecorationManager::IsDecorationVisible(const QString& decoration) const
{
  if (mitk::IRenderWindowPart::DECORATION_BORDER == decoration)
  {
    return AreAllColoredRectanglesVisible();
  }
  else if (mitk::IRenderWindowPart::DECORATION_LOGO == decoration)
  {
    return IsLogoVisible();
  }
  else if (mitk::IRenderWindowPart::DECORATION_MENU == decoration)
  {
    //return IsMenuWidgetEnabled();
  }
  else if (mitk::IRenderWindowPart::DECORATION_BACKGROUND == decoration)
  {
    return AreAllGradientBackgroundsOn();
  }
  else if (mitk::IRenderWindowPart::DECORATION_CORNER_ANNOTATION == decoration)
  {
    return AreAllCornerAnnotationsVisible();
  }

  return false;
}

QStringList QmitkMultiWidgetDecorationManager::GetDecorations() const
{
  QStringList decorations;
  decorations << mitk::IRenderWindowPart::DECORATION_BORDER << mitk::IRenderWindowPart::DECORATION_LOGO << mitk::IRenderWindowPart::DECORATION_MENU
              << mitk::IRenderWindowPart::DECORATION_BACKGROUND << mitk::IRenderWindowPart::DECORATION_CORNER_ANNOTATION;
  return decorations;
}


void QmitkMultiWidgetDecorationManager::SetupLogo(const char* path)
{
  m_LogoAnnotation->SetOpacity(0.5);
  mitk::Point2D offset;
  offset.Fill(0.03);
  m_LogoAnnotation->SetOffsetVector(offset);
  m_LogoAnnotation->SetRelativeSize(0.25);
  m_LogoAnnotation->SetCornerPosition(1);
  vtkSmartPointer<vtkImageData> vtkLogo = GetVtkLogo(path);

  SetLogo(vtkLogo);
}

void QmitkMultiWidgetDecorationManager::ShowLogo(bool show)
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetLastRenderWindowWidget();
  if (nullptr != renderWindowWidget)
  {
    m_LogoAnnotation->SetVisibility(show);
    renderWindowWidget->RequestUpdate();
    return;
  }

  MITK_ERROR << "Logo can not be shown for an unknown widget.";
}

bool QmitkMultiWidgetDecorationManager::IsLogoVisible() const
{
  return m_LogoAnnotation->IsVisible();
}

void QmitkMultiWidgetDecorationManager::SetColormap(QmitkMultiWidgetDecorationManager::Colormap colormap)
{
  switch (colormap)
  {
    case Colormap::BlackAndWhite:
    {
      FillAllGradientBackgroundColorsWithBlack();
      float white[3] = { 1.0f, 1.0f, 1.0f };
      SetAllDecorationColors(white);
      break;
    }
  }
}

void QmitkMultiWidgetDecorationManager::SetDecorationColor(const QString& widgetID, const mitk::Color& color)
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    renderWindowWidget->SetDecorationColor(color);
    return;
  }

  MITK_ERROR << "Decoration color can not be set for an unknown widget.";
}

void QmitkMultiWidgetDecorationManager::SetAllDecorationColors(const mitk::Color& color)
{
  QmitkAbstractMultiWidget::RenderWindowWidgetMap renderWindowWidgets = m_MultiWidget->GetRenderWindowWidgets();
  for (const auto& renderWindowWidget : renderWindowWidgets)
  {
    renderWindowWidget.second->SetDecorationColor(color);
  }
}

mitk::Color QmitkMultiWidgetDecorationManager::GetDecorationColor(const QString& widgetID) const
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    return renderWindowWidget->GetDecorationColor();
  }

  MITK_ERROR << "Decoration color can not be retrieved for an unknown widget. Returning black color!";
  float black[3] = { 0.0f, 0.0f, 0.0f };
  return mitk::Color(black);
}

void QmitkMultiWidgetDecorationManager::ShowColoredRectangle(const QString& widgetID, bool show)
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    renderWindowWidget->ShowColoredRectangle(show);
    return;
  }

  MITK_ERROR << "Colored rectangle can not be set for an unknown widget.";
}

void QmitkMultiWidgetDecorationManager::ShowAllColoredRectangles(bool show)
{
  QmitkAbstractMultiWidget::RenderWindowWidgetMap renderWindowWidgets = m_MultiWidget->GetRenderWindowWidgets();
  for (const auto& renderWindowWidget : renderWindowWidgets)
  {
    renderWindowWidget.second->ShowColoredRectangle(show);
  }
}

bool QmitkMultiWidgetDecorationManager::IsColoredRectangleVisible(const QString& widgetID) const
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    return renderWindowWidget->IsColoredRectangleVisible();
  }

  MITK_ERROR << "Colored rectangle visibility can not be retrieved for an unknown widget. Returning 'false'.";
  return false;
}

bool QmitkMultiWidgetDecorationManager::AreAllColoredRectanglesVisible() const
{
  QmitkAbstractMultiWidget::RenderWindowWidgetMap renderWindowWidgets = m_MultiWidget->GetRenderWindowWidgets();
  bool allTrue = true;
  for (const auto& renderWindowWidget : renderWindowWidgets)
  {
    allTrue = allTrue && renderWindowWidget.second->IsColoredRectangleVisible();
  }

  return allTrue;
}

void QmitkMultiWidgetDecorationManager::SetGradientBackgroundColors(const mitk::Color& upper, const mitk::Color& lower, const QString& widgetID)
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    renderWindowWidget->SetGradientBackgroundColors(upper, lower);
    return;
  }

  MITK_ERROR << "Background color gradient can not be set for an unknown widget.";
}

void QmitkMultiWidgetDecorationManager::SetAllGradientBackgroundColors(const mitk::Color& upper, const mitk::Color& lower)
{
  QmitkAbstractMultiWidget::RenderWindowWidgetMap renderWindowWidgets = m_MultiWidget->GetRenderWindowWidgets();
  for (const auto& renderWindowWidget : renderWindowWidgets)
  {
    renderWindowWidget.second->SetGradientBackgroundColors(upper, lower);
  }
}

void QmitkMultiWidgetDecorationManager::FillAllGradientBackgroundColorsWithBlack()
{
  float black[3] = { 0.0f, 0.0f, 0.0f };
  SetAllGradientBackgroundColors(black, black);
}

void QmitkMultiWidgetDecorationManager::ShowGradientBackground(const QString& widgetID, bool show)
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    renderWindowWidget->ShowGradientBackground(show);
    return;
  }

  MITK_ERROR << "Background color gradient can not be shown for an unknown widget.";
}

void QmitkMultiWidgetDecorationManager::ShowAllGradientBackgrounds(bool show)
{
  QmitkAbstractMultiWidget::RenderWindowWidgetMap renderWindowWidgets = m_MultiWidget->GetRenderWindowWidgets();
  for (const auto& renderWindowWidget : renderWindowWidgets)
  {
    renderWindowWidget.second->ShowGradientBackground(show);
  }
}

std::pair<mitk::Color, mitk::Color> QmitkMultiWidgetDecorationManager::GetGradientBackgroundColors(const QString& widgetID) const
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    return renderWindowWidget->GetGradientBackgroundColors();
  }

  MITK_ERROR << "Background color gradient can not be retrieved for an unknown widget. Returning black color pair.";
  float black[3] = { 0.0f, 0.0f, 0.0f };
  return std::make_pair(mitk::Color(black), mitk::Color(black));
}

bool QmitkMultiWidgetDecorationManager::IsGradientBackgroundOn(const QString& widgetID) const
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    return renderWindowWidget->IsGradientBackgroundOn();
  }

  MITK_ERROR << "Background color gradient flag can not be retrieved for an unknown widget. Returning 'false'.";
  return false;
}

bool QmitkMultiWidgetDecorationManager::AreAllGradientBackgroundsOn() const
{
  QmitkAbstractMultiWidget::RenderWindowWidgetMap renderWindowWidgets = m_MultiWidget->GetRenderWindowWidgets();
  bool allTrue = true;
  for (const auto& renderWindowWidget : renderWindowWidgets)
  {
    allTrue = allTrue && renderWindowWidget.second->IsGradientBackgroundOn();
  }

  return allTrue;
}

void QmitkMultiWidgetDecorationManager::SetCornerAnnotationText(const QString& widgetID, const std::string& cornerAnnotation)
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    renderWindowWidget->SetCornerAnnotationText(cornerAnnotation);
    return;
  }

  MITK_ERROR << "Corner annotation text can not be retrieved for an unknown widget.";
}

std::string QmitkMultiWidgetDecorationManager::GetCornerAnnotationText(const QString& widgetID) const
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    return renderWindowWidget->GetCornerAnnotationText();
  }

  MITK_ERROR << "Corner annotation text can not be retrieved for an unknown widget.";
  return "";
}

void QmitkMultiWidgetDecorationManager::ShowCornerAnnotation(const QString& widgetID, bool show)
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    renderWindowWidget->ShowCornerAnnotation(show);
    return;
  }

  MITK_ERROR << "Corner annotation can not be set for an unknown widget.";
}

void QmitkMultiWidgetDecorationManager::ShowAllCornerAnnotations(bool show)
{
  QmitkAbstractMultiWidget::RenderWindowWidgetMap renderWindowWidgets = m_MultiWidget->GetRenderWindowWidgets();
  for (const auto& renderWindowWidget : renderWindowWidgets)
  {
    renderWindowWidget.second->ShowCornerAnnotation(show);
  }
}

bool QmitkMultiWidgetDecorationManager::IsCornerAnnotationVisible(const QString& widgetID) const
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetRenderWindowWidget(widgetID);
  if (nullptr != renderWindowWidget)
  {
    return renderWindowWidget->IsCornerAnnotationVisible();
  }

  MITK_ERROR << "Corner annotation visibility can not be retrieved for an unknown widget. Returning 'false'.";
  return false;
}

bool QmitkMultiWidgetDecorationManager::AreAllCornerAnnotationsVisible() const
{
  QmitkAbstractMultiWidget::RenderWindowWidgetMap renderWindowWidgets = m_MultiWidget->GetRenderWindowWidgets();
  bool allTrue = true;
  for (const auto& renderWindowWidget : renderWindowWidgets)
  {
    allTrue = allTrue && renderWindowWidget.second->IsCornerAnnotationVisible();
  }

  return allTrue;
}

//////////////////////////////////////////////////////////////////////////
// PRIVATE
//////////////////////////////////////////////////////////////////////////
vtkSmartPointer<vtkImageData> QmitkMultiWidgetDecorationManager::GetVtkLogo(const char* path)
{
  QImage* qimage = new QImage(path);
  vtkSmartPointer<vtkQImageToImageSource> qImageToVtk;
  qImageToVtk = vtkSmartPointer<vtkQImageToImageSource>::New();

  qImageToVtk->SetQImage(qimage);
  qImageToVtk->Update();
  vtkSmartPointer<vtkImageData> vtkLogo = qImageToVtk->GetOutput();
  return vtkLogo;
}

void QmitkMultiWidgetDecorationManager::SetLogo(vtkSmartPointer<vtkImageData> vtkLogo)
{
  std::shared_ptr<QmitkRenderWindowWidget> renderWindowWidget = m_MultiWidget->GetLastRenderWindowWidget();
  if (nullptr != renderWindowWidget && m_LogoAnnotation.IsNotNull())
  {
    mitk::ManualPlacementAnnotationRenderer::AddAnnotation(m_LogoAnnotation.GetPointer(), renderWindowWidget->GetRenderWindow()->GetRenderer());
    m_LogoAnnotation->SetLogoImage(vtkLogo);
    mitk::BaseRenderer *renderer = mitk::BaseRenderer::GetInstance(renderWindowWidget->GetRenderWindow()->GetVtkRenderWindow());
    m_LogoAnnotation->Update(renderer);
    renderWindowWidget->RequestUpdate();
    return;
  }

  MITK_ERROR << "Logo can not be set for an unknown widget.";
}
