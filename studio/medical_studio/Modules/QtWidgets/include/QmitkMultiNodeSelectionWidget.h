/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkMultiNodeSelectionWidget_h
#define QmitkMultiNodeSelectionWidget_h

#include <MitkQtWidgetsExports.h>

#include <ui_QmitkMultiNodeSelectionWidget.h>

#include <mitkDataStorage.h>
#include <mitkWeakPointer.h>
#include <mitkNodePredicateBase.h>

#include <QmitkAbstractNodeSelectionWidget.h>
#include <QmitkSimpleTextOverlayWidget.h>

class QmitkAbstractDataStorageModel;

/**
* @class QmitkMultiNodeSelectionWidget
* @brief Widget that allows to perform and represents a multiple node selection.
*/
class MITKQTWIDGETS_EXPORT QmitkMultiNodeSelectionWidget : public QmitkAbstractNodeSelectionWidget
{
  Q_OBJECT

public:
  explicit QmitkMultiNodeSelectionWidget(QWidget* parent = nullptr);

  using NodeList = QmitkAbstractNodeSelectionWidget::NodeList;

  /**
  * @brief Helper function that is used to check the given selection for consistency.
  *        Returning an empty string assumes that everything is alright and the selection
  *        is valid. If the string is not empty, the content of the string will be used
  *        as error message in the overlay to indicate the problem.
  */
  using SelectionCheckFunctionType = std::function<std::string(const NodeList &)>;
  /**
  * @brief A selection check function can be set. If set the widget uses this function to
  *        check the made/set selection. If the selection is valid, everything is fine.
  *        If selection is indicated as invalid, it will not be communicated by the widget
  *        (no signal emission).
  */
  void SetSelectionCheckFunction(const SelectionCheckFunctionType &checkFunction);

  /** Returns if the current internal selection is violating the current check function, if set.*/
  bool CurrentSelectionViolatesCheckFunction() const;

Q_SIGNALS:
  void DialogClosed();

public Q_SLOTS:
  void OnEditSelection();

protected Q_SLOTS:
  void OnClearSelection(const mitk::DataNode* node);

protected:
  void changeEvent(QEvent *event) override;

  void UpdateInfo() override;
  void OnInternalSelectionChanged() override;

  bool AllowEmissionOfSelection(const NodeList& emissionCandidates) const override;

  QmitkSimpleTextOverlayWidget* m_Overlay;

  SelectionCheckFunctionType m_CheckFunction;
  mutable std::string m_CheckResponse;

  Ui_QmitkMultiNodeSelectionWidget m_Controls;
};

#endif
