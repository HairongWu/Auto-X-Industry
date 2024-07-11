/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkDicomInspectorView_h
#define QmitkDicomInspectorView_h

#include "ui_QmitkDicomInspectorViewControls.h"

// Blueberry
#include <berryIPartListener.h>

// mitk DICOM module
#include <mitkDICOMTag.h>
#include <mitkDICOMProperty.h>

// mitk gui common plugin
#include <mitkIRenderWindowPartListener.h>

// mitk gui qt common plugin
#include <QmitkAbstractView.h>
#include <QmitkSelectionServiceConnector.h>
#include <QmitkSliceNavigationListener.h>

/**
 *	@brief	View class to inspect all DICOM tags available for the data of a node.
 */
class QmitkDicomInspectorView : public QmitkAbstractView, public mitk::IRenderWindowPartListener
{

  Q_OBJECT

public:

  QmitkDicomInspectorView();
  ~QmitkDicomInspectorView() override;

  static const std::string VIEW_ID;

  void SetFocus() override { };

  void RenderWindowPartActivated(mitk::IRenderWindowPart* renderWindowPart) override;
  void RenderWindowPartDeactivated(mitk::IRenderWindowPart* renderWindowPart) override;

protected:

  void CreateQtPartControl(QWidget* parent) override;

  /** @brief Called by the selection widget when the selection has changed.*/
  void OnCurrentSelectionChanged(QList<mitk::DataNode::Pointer> nodes);

  /** @brief Sets m_currentSelectedPosition to the current selection and validates if this position is valid
   * for the input image of the currently selected fit. If it is valid, m_validSelectedPosition is set to true.
   * If the fit, his input image or geometry is not specified, it will also handled as invalid.*/
  void ValidateAndSetCurrentPosition();

protected slots:
  void OnSliceChanged();

private:

  void RenderTable();

  /** (Re-)initializes the headers of the data table.*/
  void UpdateData();
  void UpdateLabels();

  void SetAsSelectionListener(bool checked);

  Ui::QmitkDicomInspectorViewControls m_Controls;
  mitk::IRenderWindowPart* m_RenderWindowPart;

  std::unique_ptr<QmitkSelectionServiceConnector> m_SelectionServiceConnector;

  QmitkSliceNavigationListener m_SliceNavigationListener;

  /** @brief Currently selected node for the DICOM information.*/
  mitk::DataNode::ConstPointer m_SelectedNode;
  /** @brief Base data of the currently selected node.*/
  mitk::BaseData::ConstPointer m_SelectedData;

  /** @brief Valid selected position in the inspector.*/
  mitk::Point3D m_SelectedPosition;
  /** @brief Indicates if the currently selected position is valid for the currently selected data.*/
  bool m_ValidSelectedPosition;

  mitk::TimePointType m_SelectedTimePoint;

  itk::IndexValueType m_CurrentSelectedZSlice;

  /*************************************/
  /* Members for visualizing the model */

  itk::TimeStamp m_SelectedNodeTime;
  itk::TimeStamp m_CurrentPositionTime;

  /**Helper structure to manage the registered observer events.*/
  struct TagInfo
  {
    mitk::DICOMTag tag;
    mitk::DICOMProperty::ConstPointer prop;

    TagInfo(const mitk::DICOMTag& aTag, mitk::DICOMProperty::ConstPointer aProp)
      : tag(aTag)
      , prop(aProp)
    {
    };
  };

  typedef std::map<std::string, TagInfo> TagMapType;
  TagMapType m_Tags;

};

#endif
