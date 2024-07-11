/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef QmitkFitParameterWidget_h
#define QmitkFitParameterWidget_h

#include "mitkModelFitInfo.h"
#include "mitkPointSet.h"

#include "MitkModelFitUIExports.h"

#include "ui_QmitkFitParameterWidget.h"
#include <QWidget>


class QmitkFitParameterModel;

/**
* \class QmitkFitParameterWidget
* Widget that displays the parameters of all set ModelFitInfo instances for all given
* world coordinate points.
* In addition it allows to transfer this information as CSV into the clipboard or a file.
*/
class MITKMODELFITUI_EXPORT QmitkFitParameterWidget : public QWidget
{
  Q_OBJECT

public:
  using FitVectorType = std::vector<mitk::modelFit::ModelFitInfo::ConstPointer>;

  QmitkFitParameterWidget(QWidget* parent = nullptr);
  ~QmitkFitParameterWidget() override;

  const FitVectorType& getFits() const;

  mitk::Point3D getCurrentPosition() const;

  const mitk::PointSet* getPositionBookmarks() const;

public Q_SLOTS:

  void setFits(const FitVectorType& fits);

  void setCurrentPosition(const mitk::Point3D& currentPos);

  void setPositionBookmarks(const mitk::PointSet* bookmarks);

protected Q_SLOTS:
  void OnExportClicked() const;

  /** @brief Saves the results table to clipboard */
  void OnClipboardResultsButtonClicked() const;

protected:
  std::string streamModelToString() const;

  QmitkFitParameterModel * m_InternalModel;

  Ui::QmitkFitParameterWidget m_Controls;

};

/** Helper function to sanitize strings before used in a csv export
 Moved to header in order to be reusabel for other ModelFitUI widgets.*/
std::string SanatizeString(std::string str);

#endif
