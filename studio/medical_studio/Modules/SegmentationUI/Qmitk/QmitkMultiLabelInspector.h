/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkMultiLabelInspector_h
#define QmitkMultiLabelInspector_h

#include <MitkSegmentationUIExports.h>
#include <mitkWeakPointer.h>
#include <mitkLabelSetImage.h>
#include <mitkDataNode.h>
#include <mitkLabelHighlightGuard.h>

#include <QWidget>
#include <QItemSelectionModel>

class QmitkMultiLabelTreeModel;
class QStyledItemDelegate;
class QWidgetAction;

namespace Ui
{
  class QmitkMultiLabelInspector;
}

/*
* @brief This is an inspector that offers a tree view on the labels and groups of a MultiLabelSegmentation instance.
* It also allows some manipulation operations an the labels/groups according to the UI/selection state.
*/
class MITKSEGMENTATIONUI_EXPORT QmitkMultiLabelInspector : public QWidget
{
  Q_OBJECT

public:
  QmitkMultiLabelInspector(QWidget* parent = nullptr);
  ~QmitkMultiLabelInspector();

  bool GetMultiSelectionMode() const;

  bool GetAllowVisibilityModification() const;
  bool GetAllowLockModification() const;
  bool GetAllowLabelModification() const;
  /** Indicates if the inspector is currently modifying the model/segmentation.
   Thus as long as the manipulation is ongoing, one should assume the model to be in an invalid state.*/
  bool GetModelManipulationOngoing() const;

  using LabelValueType = mitk::LabelSetImage::LabelValueType;
  using LabelValueVectorType = mitk::LabelSetImage::LabelValueVectorType;

  /**
  * @brief Retrieve the currently selected labels (equals the last CurrentSelectionChanged values).
  */
  LabelValueVectorType GetSelectedLabels() const;

  /** @brief Returns the label that currently has the focus in the tree view.
   *
   * The focus is indicated by QTreeView::currentIndex, thus the mouse is over it and it has a dashed border line.
   *
   * The current label must not equal the selected label(s). If the mouse is not hovering above a label
   * (label class or instance item), the method will return nullptr.
   */
  mitk::Label* GetCurrentLabel() const;

  enum class IndexLevelType
  {
    Group,
    LabelClass,
    LabelInstance
  };

  /** @brief Returns the level of the index that currently has the focus in the tree view.
   *
   * The focus is indicated by QTreeView::currentIndex, thus the mouse is over it and it has a dashed border line.
   */
  IndexLevelType GetCurrentLevelType() const;

  /** @brief Returns all label values that are currently affected.
   *
   * Affected means that these labels (including the one returned by GetCurrentLabel) are in the subtree of the tree
   * view element that currently has the focus (indicated by QTreeView::currentIndex, thus the mouse is over it and
   * it has a dashed border line.
   */
  LabelValueVectorType GetCurrentlyAffactedLabelInstances() const;

  /** @brief Returns the values of all label instances that are of the same label (class) like the first selected label instance.
   *
   * If no label is selected an empty vector will be returned.
   */
  LabelValueVectorType GetLabelInstancesOfSelectedFirstLabel() const;

Q_SIGNALS:
  /**
  * @brief A signal that will be emitted if the selected labels change.
  *
  * @param labels A list of label values that are now selected.
  */
  void CurrentSelectionChanged(LabelValueVectorType labels) const;

  /**
  * @brief A signal that will be emitted if the user has requested to "go to" a certain label.
  *
  * Going to a label would be e.g. to focus the renderwindows on the centroid of the label.
  * @param label The label that should be focused.
  * @param point in World coordinate that should be focused.
  */
  void GoToLabel(LabelValueType label, const mitk::Point3D& point) const;

  /** @brief Signal that is emitted, if a label should be (re)named and default
  * label naming is deactivated.
  *
  * The instance for which a new name is requested is passed with the signal.
  * @param label Pointer to the instance that needs a (new) name.
  * @param rename Indicates if it is a renaming or naming of a new label.
  * @param [out] canceled Indicating if the request was canceled by the used.
  */
  void LabelRenameRequested(mitk::Label* label, bool rename, bool& canceled) const;

  /** @brief Signal that is emitted, if the model was updated (e.g. by a delete or add operation).*/
  void ModelUpdated() const;

  /** @brief Signal is emitted, if the segmentation is changed that is observed by the inspector.*/
  void SegmentationChanged() const;

public Q_SLOTS:

  /**
  * @brief Transform a list of label values into the new selection of the inspector.
  * @param selectedLabels A list of selected label values.
  * @remark Using this method to select labels will not trigger the CurrentSelectionChanged signal. Observers
  * should regard that to avoid signal loops.
  */
  void SetSelectedLabels(const LabelValueVectorType& selectedLabels);
  /**
  * @brief The passed label will be used as new selection in the widget
  * @param selectedLabel Value of the selected label.
  * @remark Using this method to select labels will not trigger the CurrentSelectionChanged signal. Observers
  * should regard that to avoid signal loops.
  */
  void SetSelectedLabel(mitk::LabelSetImage::LabelValueType selectedLabel);

  /** @brief Sets the segmentation that will be used and monitored by the widget.
  * @param segmentation      A pointer to the segmentation to set.
  * @remark You cannot set the segmentation directly if a segmentation node is
  * also set. Reset the node (nullptr) if you want to change to direct segmentation
  * setting.
  * @pre Segmentation node is nullptr.
  */
  void SetMultiLabelSegmentation(mitk::LabelSetImage* segmentation);
  mitk::LabelSetImage* GetMultiLabelSegmentation() const;

  /**
  * @brief Sets the segmentation node that will be used /monitored by the widget.
  *
  * @param node A pointer to the segmentation node.
  * @remark If not set some features (e.g. highlighting in render windows) of the inspectors
  * are not active.
  * @remark Currently it is also needed to circumvent the fact that
  * modification of data does not directly trigger modification of the
  * node (see T27307).
  */
  void SetMultiLabelNode(mitk::DataNode* node);
  mitk::DataNode* GetMultiLabelNode() const;

  void SetMultiSelectionMode(bool multiMode);

  void SetAllowVisibilityModification(bool visiblityMod);
  void SetAllowLockModification(bool lockMod);
  void SetAllowLabelModification(bool labelMod);

  void SetDefaultLabelNaming(bool defaultLabelNaming);

  /** @brief Adds an instance of the same label/class like the first label instance
  * indicated by GetSelectedLabels() to the segmentation.
  *
  * This new label instance is returned by the function. If the inspector has no selected label,
  * no new instance will be generated and nullptr will be returned.
  *
  * @remark The new label instance is a clone of the selected label instance.
  * Therefore all properties but the LabelValue will be the same.
  *
  * @pre AllowLabeModification must be set to true.
  */
  mitk::Label* AddNewLabelInstance();

  /** @brief Adds a new label to the segmentation.
  * Depending on the settings the name of
  * the label will be either default generated or the rename delegate will be used. The label
  * will be added to the same group as the first currently selected label.
  *
  * @pre AllowLabeModification must be set to true.*/
  mitk::Label* AddNewLabel();

  /** @brief Removes the first currently selected label instance of the segmentation.
  * If no label is selected
  * nothing will happen.
  *
  * @pre AllowLabeModification must be set to true.*/
  void DeleteLabelInstance();

  /** @brief Delete the first currently selected label and all its instances of the segmentation.
  * If no label is selected
  * nothing will happen.
  *
  * @pre AllowLabeModification must be set to true.*/
  void DeleteLabel();

  /** @brief Adds a new group with a new label to segmentation.
  *
  * @pre AllowLabeModification must be set to true.*/
  mitk::Label* AddNewGroup();

  /** @brief Removes the group of the first currently selected label of the segmentation.
  *If no label is selected nothing will happen.
  *
  * @pre AllowLabeModification must be set to true.*/
  void RemoveGroup();

  void SetVisibilityOfAffectedLabels(bool visible) const;
  void SetLockOfAffectedLabels(bool visible) const;

protected:
  void Initialize();
  void OnModelReset();
  void OnDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight,
    const QList<int>& roles = QList<int>());

  QmitkMultiLabelTreeModel* m_Model;
  mitk::LabelSetImage::Pointer m_Segmentation;

  LabelValueVectorType m_LastValidSelectedLabels;
  QStyledItemDelegate* m_LockItemDelegate;
  QStyledItemDelegate* m_ColorItemDelegate;
  QStyledItemDelegate* m_VisibilityItemDelegate;

  Ui::QmitkMultiLabelInspector* m_Controls;

  LabelValueVectorType GetSelectedLabelsFromSelectionModel() const;
  void UpdateSelectionModel(const LabelValueVectorType& selectedLabels);

  /** @brief Helper that returns the label object (if multiple labels are selected the first).
  */
  mitk::Label* GetFirstSelectedLabelObject() const;

  mitk::Label* AddNewLabelInternal(const mitk::LabelSetImage::GroupIndexType& containingGroup);

  /**@brief Adds an instance of the same label/class like the passed label value
  */
  mitk::Label* AddNewLabelInstanceInternal(mitk::Label* templateLabel);

  void RemoveGroupInternal(const mitk::LabelSetImage::GroupIndexType& groupID);
  void DeleteLabelInternal(const LabelValueVectorType& labelValues);

  void keyPressEvent(QKeyEvent* event) override;
  void keyReleaseEvent(QKeyEvent* event) override;

private Q_SLOTS:
  /** @brief Transform a labels selection into a data node list and emit the 'CurrentSelectionChanged'-signal.
  *
  * The function adds the selected nodes from the original selection that could not be modified, if
  * m_SelectOnlyVisibleNodes is false.
  * This slot is internally connected to the 'selectionChanged'-signal of the selection model of the private member item view.
  *
  * @param selected	The newly selected items.
  * @param deselected	The newly deselected items.
  */
  void OnChangeModelSelection(const QItemSelection& selected, const QItemSelection& deselected);

  void OnContextMenuRequested(const QPoint&);

  void OnAddLabel();
  void OnAddLabelInstance();
  void OnDeleteGroup();
  void OnDeleteAffectedLabel();
  void OnDeleteLabels(bool);
  void OnClearLabels(bool);
  void OnMergeLabels(bool);

  void OnRenameLabel(bool);
  void OnClearLabel(bool);

  void OnUnlockAffectedLabels();
  void OnLockAffectedLabels();

  void OnSetAffectedLabelsVisible();
  void OnSetAffectedLabelsInvisible();
  void OnSetOnlyActiveLabelVisible(bool);

  void OnItemDoubleClicked(const QModelIndex& index);

  void WaitCursorOn() const;
  void WaitCursorOff() const;
  void RestoreOverrideCursor() const;

  void PrepareGoToLabel(LabelValueType labelID) const;

  void OnEntered(const QModelIndex& index);
  void OnMouseLeave();

  QWidgetAction* CreateOpacityAction();

private:
  bool m_ShowVisibility = true;
  bool m_ShowLock = true;
  bool m_ShowOther = false;

  /** @brief Indicates if the context menu allows changes in visibility.
  *
  * Visibility includes also color
  */
  bool m_AllowVisibilityModification = true;

  /** @brief Indicates if the context menu allows changes in lock state.
  */
  bool m_AllowLockModification = true;

  /** @brief Indicates if the context menu allows label modifications (adding, removing, renaming ...)
  */
  bool m_AllowLabelModification = false;

  bool m_DefaultLabelNaming = true;

  bool m_ModelManipulationOngoing = false;

  bool m_AboutToShowContextMenu = false;
  mitk::DataNode::Pointer m_SegmentationNode;
  unsigned long m_SegmentationNodeDataMTime;
  mitk::ITKEventObserverGuard m_SegmentationObserver;
  mitk::LabelHighlightGuard m_LabelHighlightGuard;
};

#endif
