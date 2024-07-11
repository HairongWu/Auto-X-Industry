/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef GenericDataFittingView_h
#define GenericDataFittingView_h

#include <QString>

#include "QmitkAbstractView.h"
#include "itkCommand.h"

#include "ui_GenericDataFittingViewControls.h"
#include "mitkModelBase.h"
#include "QmitkParameterFitBackgroundJob.h"
#include "mitkModelFitResultHelper.h"
#include "mitkModelFactoryBase.h"
#include "mitkLevenbergMarquardtModelFitFunctor.h"
#include "mitkSimpleBarrierConstraintChecker.h"

#include <mitkNodePredicateBase.h>

/*!
*	@brief Plugin for generic dynamic image data fitting
*/
class GenericDataFittingView : public QmitkAbstractView
{
  Q_OBJECT

public:

  /*! @brief The view's unique ID - required by MITK */
  static const std::string VIEW_ID;

  GenericDataFittingView();

protected slots:

  void OnModellingButtonClicked();

  void OnJobFinished();
  void OnJobError(QString err);
  void OnJobResultsAreAvailable(mitk::modelFit::ModelFitResultNodeVectorType results,
                                const ParameterFitBackgroundJob* pJob);
  void OnJobProgress(double progress);
  void OnJobStatusChanged(QString info);

  void OnModellSet(int);

  void OnNrOfParamsChanged();

  /**Sets visibility and enabled state of the GUI depending on the settings and workflow state.*/
  void UpdateGUIControls();

protected:
  typedef QList<mitk::DataNode*> SelectedDataNodeVectorType;

  // Overridden base class functions

  /*!
  *	@brief					Sets up the UI controls and connects the slots and signals. Gets
  *							called by the framework to create the GUI at the right time.
  *	@param[in,out] parent	The parent QWidget, as this class itself is not a QWidget
  *							subclass.
  */
  void CreateQtPartControl(QWidget* parent) override;

  /*!
  *	@brief	Sets the focus to the plot curve button. Gets called by the framework to set the
  *			focus on the right widget.
  */
  void SetFocus() override;

  template <typename TParameterizer>
  void GenerateModelFit_ROIBased(mitk::modelFit::ModelFitInfo::Pointer& modelFitInfo,
                                 mitk::ParameterFitImageGeneratorBase::Pointer& generator);

  template <typename TParameterizer>
  void GenerateModelFit_PixelBased(mitk::modelFit::ModelFitInfo::Pointer& modelFitInfo,
                                   mitk::ParameterFitImageGeneratorBase::Pointer& generator);

  /** Helper function that configures the initial parameter strategy of a parameterizer
   according to the settings of the GUI.*/
  void ConfigureInitialParametersOfParameterizer(mitk::ModelParameterizerBase* parameterizer) const;

  void PrepareFitConfiguration();

  bool IsGenericParamFactorySelected() const;

  /*! Starts the fitting job with the passed generator and session info*/
  void DoFit(const mitk::modelFit::ModelFitInfo* fitSession,
             mitk::ParameterFitImageGeneratorBase* generator);

  /**Checks if the settings in the GUI are valid for the chosen model.*/
  bool CheckModelSettings() const;

  void InitModelComboBox() const;

  void OnImageNodeSelectionChanged(QList<mitk::DataNode::Pointer> /*nodes*/);

  void OnMaskNodeSelectionChanged(QList<mitk::DataNode::Pointer> /*nodes*/);


  // Variables

  /*! @brief The view's UI controls */
  Ui::GeneralDataFittingViewControls m_Controls;

  /* Nodes selected by user/ui for the fit */
  mitk::DataNode::Pointer m_selectedNode;
  mitk::DataNode::Pointer m_selectedMaskNode;

  /* Images selected by user/ui for the fit */
  mitk::Image::Pointer m_selectedImage;
  mitk::Image::Pointer m_selectedMask;

  mitk::ModelFactoryBase::Pointer m_selectedModelFactory;

  mitk::SimpleBarrierConstraintChecker::Pointer m_modelConstraints;

private:

  bool m_FittingInProgress;

  typedef std::vector<mitk::ModelFactoryBase::Pointer> ModelFactoryStackType;
  ModelFactoryStackType m_FactoryStack;

  /**Helper function that generates a default fitting functor
   * default is a levenberg marquart based optimizer with all scales set to 1.0.
   * Constraint setter will be set based on the gui setting and a evaluation parameter
   * "sum of squared differences" will always be set.*/
  mitk::ModelFitFunctorBase::Pointer CreateDefaultFitFunctor(const mitk::ModelParameterizerBase*
      parameterizer) const;

  /**Returns the default fit name, derived from the current GUI settings.*/
  std::string GetDefaultFitName() const;
  /**Returns the current set name of the fit (either default name or use defined name).*/
  std::string GetFitName() const;

  mitk::NodePredicateBase::Pointer m_isValidTimeSeriesImagePredicate;

};

#endif
