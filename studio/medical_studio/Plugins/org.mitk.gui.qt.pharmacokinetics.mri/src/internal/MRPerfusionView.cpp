/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "MRPerfusionView.h"

#include "boost/tokenizer.hpp"
#include "boost/math/constants/constants.hpp"
#include <iostream>

#include "mitkWorkbenchUtil.h"

#include "mitkAterialInputFunctionGenerator.h"
#include "mitkConcentrationCurveGenerator.h"

#include <mitkDescriptivePharmacokineticBrixModelFactory.h>
#include <mitkDescriptivePharmacokineticBrixModelParameterizer.h>
#include <mitkDescriptivePharmacokineticBrixModelValueBasedParameterizer.h>
#include <mitkExtendedToftsModelFactory.h>
#include <mitkExtendedToftsModelParameterizer.h>
#include <mitkStandardToftsModelFactory.h>
#include <mitkStandardToftsModelParameterizer.h>
#include "mitkTwoCompartmentExchangeModelFactory.h"
#include "mitkTwoCompartmentExchangeModelParameterizer.h"
#include <mitkInitialParameterizationDelegateBase.h>

#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateOr.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateDimension.h>
#include "mitkNodePredicateFunction.h"
#include <mitkMultiLabelPredicateHelper.h>
#include <mitkPixelBasedParameterFitImageGenerator.h>
#include <mitkROIBasedParameterFitImageGenerator.h>
#include <mitkLevenbergMarquardtModelFitFunctor.h>
#include <mitkSumOfSquaredDifferencesFitCostFunction.h>
#include <mitkNormalizedSumOfSquaredDifferencesFitCostFunction.h>
#include <mitkSimpleBarrierConstraintChecker.h>
#include <mitkModelFitResultHelper.h>
#include <mitkImageTimeSelector.h>
#include <mitkMaskedDynamicImageStatisticsGenerator.h>
#include <mitkExtractTimeGrid.h>
#include <mitkModelFitResultRelationRule.h>
#include <mitkLabelSetImageConverter.h>

#include <QMessageBox>
#include <QThreadPool>
#include <QFileDialog>


// Includes for image casting between ITK and MITK
#include <mitkImage.h>
#include "mitkImageCast.h"
#include "mitkITKImageImport.h"
#include <itkImage.h>
#include <itkImageRegionIterator.h>




const std::string MRPerfusionView::VIEW_ID = "org.mitk.views.pharmacokinetics.mri";

inline double convertToDouble(const std::string& data)
{
  std::istringstream stepStream(data);
  stepStream.imbue(std::locale("C"));
  double value = 0.0;

  if (!(stepStream >> value) || !(stepStream.eof()))
  {
    mitkThrow() << "Cannot convert string to double. String: " << data;
  }
  return value;
}

void MRPerfusionView::SetFocus()
{
  m_Controls.btnModelling->setFocus();
}

void MRPerfusionView::CreateQtPartControl(QWidget* parent)
{
  m_Controls.setupUi(parent);

  m_Controls.btnModelling->setEnabled(false);

  this->InitModelComboBox();
  m_Controls.labelMaskInfo->hide();

  m_Controls.timeSeriesNodeSelector->SetNodePredicate(this->m_isValidTimeSeriesImagePredicate);
  m_Controls.timeSeriesNodeSelector->SetDataStorage(this->GetDataStorage());
  m_Controls.timeSeriesNodeSelector->SetSelectionIsOptional(false);
  m_Controls.timeSeriesNodeSelector->SetInvalidInfo("Please select time series.");
  m_Controls.timeSeriesNodeSelector->SetAutoSelectNewNodes(true);

  m_Controls.maskNodeSelector->SetNodePredicate(this->m_IsMaskPredicate);
  m_Controls.maskNodeSelector->SetDataStorage(this->GetDataStorage());
  m_Controls.maskNodeSelector->SetSelectionIsOptional(true);
  m_Controls.maskNodeSelector->SetEmptyInfo("Please select (optional) mask.");
  connect(m_Controls.btnModelling, SIGNAL(clicked()), this, SLOT(OnModellingButtonClicked()));

  connect(m_Controls.comboModel, SIGNAL(currentIndexChanged(int)), this, SLOT(OnModellSet(int)));
  connect(m_Controls.radioPixelBased, SIGNAL(toggled(bool)), this, SLOT(UpdateGUIControls()));

  connect(m_Controls.checkMaskInfo, SIGNAL(toggled(bool)), m_Controls.labelMaskInfo,
    SLOT(setVisible(bool)));


  connect(m_Controls.timeSeriesNodeSelector, &QmitkAbstractNodeSelectionWidget::CurrentSelectionChanged, this, &MRPerfusionView::OnImageNodeSelectionChanged);
  connect(m_Controls.maskNodeSelector, &QmitkAbstractNodeSelectionWidget::CurrentSelectionChanged, this, &MRPerfusionView::OnMaskNodeSelectionChanged);


  connect(m_Controls.AIFMaskNodeSelector,
    &QmitkAbstractNodeSelectionWidget::CurrentSelectionChanged,
    this,
    &MRPerfusionView::UpdateGUIControls);

  connect(m_Controls.AIFImageNodeSelector,
    &QmitkAbstractNodeSelectionWidget::CurrentSelectionChanged,
    this,
    &MRPerfusionView::UpdateGUIControls);

  //AIF setting
  m_Controls.groupAIF->hide();
  m_Controls.btnAIFFile->setEnabled(false);
  m_Controls.btnAIFFile->setVisible(false);
  m_Controls.aifFilePath->setEnabled(false);
  m_Controls.aifFilePath->setVisible(false);
  m_Controls.radioAIFImage->setChecked(true);
  m_Controls.AIFMaskNodeSelector->SetDataStorage(this->GetDataStorage());
  m_Controls.AIFMaskNodeSelector->SetNodePredicate(m_IsMaskPredicate);
  m_Controls.AIFMaskNodeSelector->setVisible(true);
  m_Controls.AIFMaskNodeSelector->setEnabled(true);
  m_Controls.AIFMaskNodeSelector->SetAutoSelectNewNodes(true);
  m_Controls.AIFImageNodeSelector->SetDataStorage(this->GetDataStorage());
  m_Controls.AIFImageNodeSelector->SetNodePredicate(this->m_isValidTimeSeriesImagePredicate);
  m_Controls.AIFImageNodeSelector->setEnabled(false);
  m_Controls.AIFImageNodeSelector->setVisible(false);

  m_Controls.checkDedicatedAIFImage->setEnabled(true);

  m_Controls.HCLSpinBox->setValue(mitk::AterialInputFunctionGenerator::DEFAULT_HEMATOCRIT_LEVEL);

  connect(m_Controls.radioAIFImage, SIGNAL(toggled(bool)), m_Controls.AIFMaskNodeSelector, SLOT(setVisible(bool)));
  connect(m_Controls.radioAIFImage, SIGNAL(toggled(bool)), m_Controls.AIFMaskNodeSelector, SLOT(setEnabled(bool)));
  connect(m_Controls.radioAIFImage, SIGNAL(toggled(bool)), m_Controls.labelAIFMask, SLOT(setVisible(bool)));
  connect(m_Controls.radioAIFImage, SIGNAL(toggled(bool)), m_Controls.checkDedicatedAIFImage, SLOT(setVisible(bool)));
  connect(m_Controls.radioAIFImage, SIGNAL(toggled(bool)), m_Controls.checkDedicatedAIFImage, SLOT(setEnabled(bool)));
  connect(m_Controls.checkDedicatedAIFImage, SIGNAL(toggled(bool)), m_Controls.AIFImageNodeSelector, SLOT(setEnabled(bool)));
  connect(m_Controls.checkDedicatedAIFImage, SIGNAL(toggled(bool)), m_Controls.AIFImageNodeSelector, SLOT(setVisible(bool)));
  connect(m_Controls.radioAIFFile, SIGNAL(toggled(bool)), m_Controls.btnAIFFile, SLOT(setEnabled(bool)));
  connect(m_Controls.radioAIFFile, SIGNAL(toggled(bool)), m_Controls.btnAIFFile, SLOT(setVisible(bool)));
  connect(m_Controls.radioAIFFile, SIGNAL(toggled(bool)), m_Controls.aifFilePath, SLOT(setEnabled(bool)));
  connect(m_Controls.radioAIFFile, SIGNAL(toggled(bool)), m_Controls.aifFilePath, SLOT(setVisible(bool)));
  connect(m_Controls.radioAIFFile, SIGNAL(toggled(bool)), this, SLOT(UpdateGUIControls()));


  connect(m_Controls.btnAIFFile, SIGNAL(clicked()), this, SLOT(LoadAIFfromFile()));



  //Brix setting
  m_Controls.groupDescBrix->hide();
  connect(m_Controls.injectiontime, SIGNAL(valueChanged(double)), this, SLOT(UpdateGUIControls()));


  //Model fit configuration
  m_Controls.groupBox_FitConfiguration->hide();

  m_Controls.checkBox_Constraints->setEnabled(false);
  m_Controls.constraintManager->setEnabled(false);
  m_Controls.initialValuesManager->setEnabled(false);
  m_Controls.initialValuesManager->setDataStorage(this->GetDataStorage());

  connect(m_Controls.radioButton_StartParameters, SIGNAL(toggled(bool)), this, SLOT(UpdateGUIControls()));
  connect(m_Controls.checkBox_Constraints, SIGNAL(toggled(bool)), this, SLOT(UpdateGUIControls()));
  connect(m_Controls.initialValuesManager, SIGNAL(initialValuesChanged(void)), this, SLOT(UpdateGUIControls()));


  connect(m_Controls.radioButton_StartParameters, SIGNAL(toggled(bool)), m_Controls.initialValuesManager, SLOT(setEnabled(bool)));
  connect(m_Controls.checkBox_Constraints, SIGNAL(toggled(bool)), m_Controls.constraintManager, SLOT(setEnabled(bool)));
  connect(m_Controls.checkBox_Constraints, SIGNAL(toggled(bool)), m_Controls.constraintManager, SLOT(setVisible(bool)));

  //Concentration
  m_Controls.groupConcentration->hide();
  m_Controls.groupBoxEnhancement->hide();
  m_Controls.radioButtonNoConversion->setChecked(true);
  m_Controls.groupBox_T1MapviaVFA->hide();

  m_Controls.spinBox_baselineStartTimeStep->setValue(0);
  m_Controls.spinBox_baselineEndTimeStep->setValue(0);
  m_Controls.spinBox_baselineEndTimeStep->setMinimum(0);
  m_Controls.spinBox_baselineStartTimeStep->setMinimum(0);
  m_Controls.groupBox_baselineRangeSelection->hide();




  connect(m_Controls.radioButton_absoluteEnhancement, SIGNAL(toggled(bool)), this, SLOT(UpdateGUIControls()));
  connect(m_Controls.radioButton_relativeEnchancement, SIGNAL(toggled(bool)), this, SLOT(UpdateGUIControls()));
  connect(m_Controls.radioButton_absoluteEnhancement, SIGNAL(toggled(bool)), m_Controls.groupBoxEnhancement, SLOT(setVisible(bool)));
  connect(m_Controls.radioButton_absoluteEnhancement, SIGNAL(toggled(bool)), m_Controls.groupBox_baselineRangeSelection, SLOT(setVisible(bool)));
  connect(m_Controls.radioButton_relativeEnchancement, SIGNAL(toggled(bool)), m_Controls.groupBoxEnhancement, SLOT(setVisible(bool)));
  connect(m_Controls.radioButton_relativeEnchancement, SIGNAL(toggled(bool)), m_Controls.groupBox_baselineRangeSelection, SLOT(setVisible(bool)));

  connect(m_Controls.factorSpinBox, SIGNAL(valueChanged(double)), this, SLOT(UpdateGUIControls()));
  connect(m_Controls.spinBox_baselineStartTimeStep, SIGNAL(valueChanged(int)), this, SLOT(UpdateGUIControls()));
  connect(m_Controls.spinBox_baselineEndTimeStep, SIGNAL(valueChanged(int)), this, SLOT(UpdateGUIControls()));

  connect(m_Controls.radioButtonUsingT1viaVFA, SIGNAL(toggled(bool)), m_Controls.groupBox_T1MapviaVFA, SLOT(setVisible(bool)));
  connect(m_Controls.radioButtonUsingT1viaVFA, SIGNAL(toggled(bool)), m_Controls.groupBox_baselineRangeSelection, SLOT(setVisible(bool)));
  connect(m_Controls.radioButtonUsingT1viaVFA, SIGNAL(toggled(bool)), this, SLOT(UpdateGUIControls()));
  connect(m_Controls.FlipangleSpinBox, SIGNAL(valueChanged(double)), this, SLOT(UpdateGUIControls()));
  connect(m_Controls.RelaxivitySpinBox, SIGNAL(valueChanged(double)), this, SLOT(UpdateGUIControls()));
  connect(m_Controls.TRSpinBox, SIGNAL(valueChanged(double)), this, SLOT(UpdateGUIControls()));

  m_Controls.PDWImageNodeSelector->SetNodePredicate(m_isValidPDWImagePredicate);
  m_Controls.PDWImageNodeSelector->SetDataStorage(this->GetDataStorage());
  m_Controls.PDWImageNodeSelector->SetInvalidInfo("Please select PDW Image.");
  m_Controls.PDWImageNodeSelector->setEnabled(false);

  connect(m_Controls.radioButtonUsingT1viaVFA, SIGNAL(toggled(bool)), m_Controls.PDWImageNodeSelector, SLOT(setEnabled(bool)));

  UpdateGUIControls();
}




void MRPerfusionView::UpdateGUIControls()
{
  m_Controls.lineFitName->setPlaceholderText(QString::fromStdString(this->GetDefaultFitName()));
  m_Controls.lineFitName->setEnabled(!m_FittingInProgress);

  m_Controls.checkBox_Constraints->setEnabled(m_modelConstraints.IsNotNull());

  bool isDescBrixFactory = dynamic_cast<mitk::DescriptivePharmacokineticBrixModelFactory*>
                           (m_selectedModelFactory.GetPointer()) != nullptr;
  bool isToftsFactory = dynamic_cast<mitk::StandardToftsModelFactory*>
                        (m_selectedModelFactory.GetPointer()) != nullptr ||
                         dynamic_cast<mitk::ExtendedToftsModelFactory*>
                                  (m_selectedModelFactory.GetPointer()) != nullptr;
  bool is2CXMFactory = dynamic_cast<mitk::TwoCompartmentExchangeModelFactory*>
                       (m_selectedModelFactory.GetPointer()) != nullptr;




  m_Controls.groupAIF->setVisible(isToftsFactory || is2CXMFactory);
  m_Controls.groupDescBrix->setVisible(isDescBrixFactory);
  if (isDescBrixFactory)
  {
    m_Controls.toolboxConfiguration->setItemEnabled(2, false);
  }
  else
  {
    m_Controls.toolboxConfiguration->setItemEnabled(2, true);
  }
  m_Controls.groupConcentration->setVisible(isToftsFactory || is2CXMFactory );
  m_Controls.AIFImageNodeSelector->setVisible(!m_Controls.radioAIFFile->isChecked());
  m_Controls.AIFImageNodeSelector->setVisible(m_Controls.radioAIFImage->isChecked() && m_Controls.checkDedicatedAIFImage->isChecked());

  m_Controls.groupBox_FitConfiguration->setVisible(m_selectedModelFactory);

  m_Controls.groupBox->setEnabled(!m_FittingInProgress);
  m_Controls.comboModel->setEnabled(!m_FittingInProgress);
  m_Controls.groupAIF->setEnabled(!m_FittingInProgress);
  m_Controls.groupDescBrix->setEnabled(!m_FittingInProgress);
  m_Controls.groupConcentration->setEnabled(!m_FittingInProgress);
  m_Controls.groupBox_FitConfiguration->setEnabled(!m_FittingInProgress);


  m_Controls.radioROIbased->setEnabled(m_selectedMask.IsNotNull());

  m_Controls.btnModelling->setEnabled(m_selectedImage.IsNotNull()
                                      && m_selectedModelFactory.IsNotNull() && !m_FittingInProgress && CheckModelSettings());

  m_Controls.spinBox_baselineStartTimeStep->setEnabled( m_Controls.radioButton_absoluteEnhancement->isChecked() || m_Controls.radioButton_relativeEnchancement->isChecked() || m_Controls.radioButtonUsingT1viaVFA->isChecked());
  m_Controls.spinBox_baselineEndTimeStep->setEnabled(m_Controls.radioButton_absoluteEnhancement->isChecked() || m_Controls.radioButton_relativeEnchancement->isChecked() || m_Controls.radioButtonUsingT1viaVFA->isChecked());


}

void MRPerfusionView::OnModellSet(int index)
{
  m_selectedModelFactory = nullptr;

  if (index > 0)
  {
    if (static_cast<ModelFactoryStackType::size_type>(index) <= m_FactoryStack.size() )
    {
        m_selectedModelFactory = m_FactoryStack[index - 1];
    }
    else
    {
        MITK_WARN << "Invalid model index. Index outside of the factory stack. Factory stack size: "<< m_FactoryStack.size() << "; invalid index: "<< index;
    }
  }

  if (m_selectedModelFactory)
  {
    this->m_modelConstraints = dynamic_cast<mitk::SimpleBarrierConstraintChecker*>
                               (m_selectedModelFactory->CreateDefaultConstraints().GetPointer());

    m_Controls.initialValuesManager->setInitialValues(m_selectedModelFactory->GetParameterNames(),
        m_selectedModelFactory->GetDefaultInitialParameterization());

    if (this->m_modelConstraints.IsNull())
    {
      this->m_modelConstraints = mitk::SimpleBarrierConstraintChecker::New();
    }

    m_Controls.constraintManager->setChecker(this->m_modelConstraints,
        this->m_selectedModelFactory->GetParameterNames());

  }

  UpdateGUIControls();
}

std::string MRPerfusionView::GetFitName() const
{
  std::string fitName = m_Controls.lineFitName->text().toStdString();
  if (fitName.empty())
  {
    fitName = m_Controls.lineFitName->placeholderText().toStdString();
  }
  return fitName;
}

std::string MRPerfusionView::GetDefaultFitName() const
{
    std::string defaultName = "undefined model";

    if (this->m_selectedModelFactory.IsNotNull())
    {
        defaultName = this->m_selectedModelFactory->GetClassID();
    }

    if (this->m_Controls.radioPixelBased->isChecked())
    {
        defaultName += "_pixel";
    }
    else
    {
        defaultName += "_roi";
    }

    return defaultName;
}

void MRPerfusionView::OnModellingButtonClicked()
{
  //check if all static parameters set
  if (m_selectedModelFactory.IsNotNull() && CheckModelSettings())
  {
    m_HasGeneratedNewInput = false;
    m_HasGeneratedNewInputAIF = false;

    mitk::ParameterFitImageGeneratorBase::Pointer generator = nullptr;
    mitk::modelFit::ModelFitInfo::Pointer fitSession = nullptr;

    bool isDescBrixFactory = dynamic_cast<mitk::DescriptivePharmacokineticBrixModelFactory*>
                             (m_selectedModelFactory.GetPointer()) != nullptr;
    bool isExtToftsFactory = dynamic_cast<mitk::ExtendedToftsModelFactory*>
                          (m_selectedModelFactory.GetPointer()) != nullptr;
    bool isStanToftsFactory = dynamic_cast<mitk::StandardToftsModelFactory*>
                          (m_selectedModelFactory.GetPointer()) != nullptr;
    bool is2CXMFactory = dynamic_cast<mitk::TwoCompartmentExchangeModelFactory*>
                         (m_selectedModelFactory.GetPointer()) != nullptr;

    if (isDescBrixFactory)
    {
      if (this->m_Controls.radioPixelBased->isChecked())
      {
        GenerateDescriptiveBrixModel_PixelBased(fitSession, generator);
      }
      else
      {
        GenerateDescriptiveBrixModel_ROIBased(fitSession, generator);
      }
    }
    else if (isStanToftsFactory)
    {
      if (this->m_Controls.radioPixelBased->isChecked())
      {
        GenerateAIFbasedModelFit_PixelBased<mitk::StandardToftsModelParameterizer>(fitSession, generator);
      }
      else
      {
        GenerateAIFbasedModelFit_ROIBased<mitk::StandardToftsModelParameterizer>(fitSession, generator);
      }
    }
    else if (isExtToftsFactory)
    {
      if (this->m_Controls.radioPixelBased->isChecked())
      {
        GenerateAIFbasedModelFit_PixelBased<mitk::ExtendedToftsModelParameterizer>(fitSession, generator);
      }
      else
      {
        GenerateAIFbasedModelFit_ROIBased<mitk::ExtendedToftsModelParameterizer>(fitSession, generator);
      }
    }
    else if (is2CXMFactory)
    {
      if (this->m_Controls.radioPixelBased->isChecked())
      {
        GenerateAIFbasedModelFit_PixelBased<mitk::TwoCompartmentExchangeModelParameterizer>(fitSession, generator);
      }
      else
      {
        GenerateAIFbasedModelFit_ROIBased<mitk::TwoCompartmentExchangeModelParameterizer>(fitSession, generator);
      }
    }

    //add other models with else if

    if (generator.IsNotNull() && fitSession.IsNotNull())
    {
      m_FittingInProgress = true;
      UpdateGUIControls();
      DoFit(fitSession, generator);
    }
    else
    {
      QMessageBox box;
      box.setText("Fitting error!");
      box.setInformativeText("Could not establish fitting job. Error when setting ab generator, model parameterizer or session info.");
      box.setStandardButtons(QMessageBox::Ok);
      box.setDefaultButton(QMessageBox::Ok);
      box.setIcon(QMessageBox::Warning);
      box.exec();
    }

  }
  else
  {
    QMessageBox box;
    box.setText("Static parameters for model are not set!");
    box.setInformativeText("Some static parameters, that are needed for calculation are not set and equal to zero. Modeling not possible");
    box.setStandardButtons(QMessageBox::Ok);
    box.setDefaultButton(QMessageBox::Ok);
    box.setIcon(QMessageBox::Warning);
    box.exec();
  }
}

void MRPerfusionView::OnImageNodeSelectionChanged(QList<mitk::DataNode::Pointer>/*nodes*/)
{

  if (m_Controls.timeSeriesNodeSelector->GetSelectedNode().IsNotNull())
  {
    this->m_selectedNode = m_Controls.timeSeriesNodeSelector->GetSelectedNode();
    m_selectedImage = dynamic_cast<mitk::Image*>(m_selectedNode->GetData());

    if (m_selectedImage)
    {
      this->m_Controls.initialValuesManager->setReferenceImageGeometry(m_selectedImage->GetGeometry());
      m_Controls.maskNodeSelector->SetNodePredicate(mitk::GetMultiLabelSegmentationPredicate(m_selectedImage->GetGeometry()));
    }
    else
    {
      this->m_Controls.initialValuesManager->setReferenceImageGeometry(nullptr);
      m_Controls.maskNodeSelector->SetNodePredicate(mitk::GetMultiLabelSegmentationPredicate(nullptr));
    }
  }
  else
  {
    this->m_selectedNode = nullptr;
    this->m_selectedImage = nullptr;
    this->m_Controls.initialValuesManager->setReferenceImageGeometry(nullptr);
  }

  if (this->m_selectedImage.IsNotNull())
  {
    m_Controls.spinBox_baselineStartTimeStep->setMaximum((this->m_selectedImage->GetDimension(3)) - 1);
    m_Controls.spinBox_baselineEndTimeStep->setMaximum((this->m_selectedImage->GetDimension(3)) - 1);
  }

  UpdateGUIControls();
}


void MRPerfusionView::OnMaskNodeSelectionChanged(QList<mitk::DataNode::Pointer>/*nodes*/)
{
  m_selectedMaskNode = nullptr;
  m_selectedMask = nullptr;

  if (m_Controls.maskNodeSelector->GetSelectedNode().IsNotNull())
  {
    this->m_selectedMaskNode = m_Controls.maskNodeSelector->GetSelectedNode();
    auto selectedLabelSetMask = dynamic_cast<mitk::LabelSetImage*>(m_selectedMaskNode->GetData());

    if (selectedLabelSetMask != nullptr)
    {
      if (selectedLabelSetMask->GetAllLabelValues().size() > 1)
      {
        MITK_INFO << "Selected mask has multiple labels. Only use first used to mask the model fit.";
      }
      this->m_selectedMask = mitk::CreateLabelMask(selectedLabelSetMask, selectedLabelSetMask->GetAllLabelValues().front(), true);
    }


    if (this->m_selectedMask.IsNotNull() && this->m_selectedMask->GetTimeSteps() > 1)
    {
      MITK_INFO <<
        "Selected mask has multiple timesteps. Only use first timestep to mask model fit. Mask name: " <<
        m_Controls.maskNodeSelector->GetSelectedNode()->GetName();
      this->m_selectedMask = SelectImageByTimeStep(m_selectedMask, 0);

    }
  }

  if (m_selectedMask.IsNull())
  {
    this->m_Controls.radioPixelBased->setChecked(true);
  }


  UpdateGUIControls();
}


bool MRPerfusionView::CheckModelSettings() const
{
  bool ok = true;

  //check whether any model is set at all. Otherwise exit with false
  if (m_selectedModelFactory.IsNotNull())
  {
    bool isDescBrixFactory = dynamic_cast<mitk::DescriptivePharmacokineticBrixModelFactory*>
                             (m_selectedModelFactory.GetPointer()) != nullptr;
    bool isToftsFactory = dynamic_cast<mitk::StandardToftsModelFactory*>
                          (m_selectedModelFactory.GetPointer()) != nullptr||
                          dynamic_cast<mitk::ExtendedToftsModelFactory*>
                          (m_selectedModelFactory.GetPointer()) != nullptr;
    bool is2CXMFactory = dynamic_cast<mitk::TwoCompartmentExchangeModelFactory*>
                         (m_selectedModelFactory.GetPointer()) != nullptr;

    if (isDescBrixFactory)
    {
      //if all static parameters for this model are set, exit with true, Otherwise exit with false
      ok = m_Controls.injectiontime->value() > 0;
    }
    else if (isToftsFactory || is2CXMFactory)
    {
      if (this->m_Controls.radioAIFImage->isChecked())
      {
        ok = ok && m_Controls.AIFMaskNodeSelector->GetSelectedNode().IsNotNull();

        if (this->m_Controls.checkDedicatedAIFImage->isChecked())
        {
          ok = ok && m_Controls.AIFImageNodeSelector->GetSelectedNode().IsNotNull();
        }
      }
      else if (this->m_Controls.radioAIFFile->isChecked())
      {
        ok = ok && (this->AIFinputGrid.size() != 0) && (this->AIFinputFunction.size() != 0);
      }
      else
      {
        ok = false;
      }

      if (this->m_Controls.radioButton_absoluteEnhancement->isChecked()
               || this->m_Controls.radioButton_relativeEnchancement->isChecked() )
      {
        ok = ok && (m_Controls.factorSpinBox->value() > 0);
        ok = ok && CheckBaselineSelectionSettings();
      }
      else if (this->m_Controls.radioButtonUsingT1viaVFA->isChecked() )
      {
        ok = ok && (m_Controls.FlipangleSpinBox->value() > 0);
        ok = ok && (m_Controls.TRSpinBox->value() > 0);
        ok = ok && (m_Controls.RelaxivitySpinBox->value() > 0);
        ok = ok && (m_Controls.PDWImageNodeSelector->GetSelectedNode().IsNotNull());
        ok = ok && CheckBaselineSelectionSettings();
      }
      else if (this->m_Controls.radioButtonNoConversion->isChecked())
      {
        ok = ok && true;
      }
      else
      {
        ok = false;
      }

    }
    //add other models as else if and check whether all needed static parameters are set
    else
    {
      ok = false;
    }

    if (this->m_Controls.radioButton_StartParameters->isChecked() && !this->m_Controls.initialValuesManager->hasValidInitialValues())
    {
      std::string warning = "Warning. Invalid start parameters. At least one parameter as an invalid image setting as source.";
      MITK_ERROR << warning;
      m_Controls.infoBox->append(QString("<font color='red'><b>") + QString::fromStdString(warning) + QString("</b></font>"));

      ok = false;
    };
  }
  else
  {
    ok = false;
  }

  return ok;
}

bool MRPerfusionView::CheckBaselineSelectionSettings() const
{
  return m_Controls.spinBox_baselineStartTimeStep->value() <= m_Controls.spinBox_baselineEndTimeStep->value();
}

void MRPerfusionView::ConfigureInitialParametersOfParameterizer(mitk::ModelParameterizerBase*
    parameterizer) const
{
  if (m_Controls.radioButton_StartParameters->isChecked())
  {
    //use user defined initial parameters
    mitk::InitialParameterizationDelegateBase::Pointer paramDelegate = m_Controls.initialValuesManager->getInitialParametrizationDelegate();
    parameterizer->SetInitialParameterizationDelegate(paramDelegate);
  }
}

void MRPerfusionView::GenerateDescriptiveBrixModel_PixelBased(mitk::modelFit::ModelFitInfo::Pointer&
    modelFitInfo, mitk::ParameterFitImageGeneratorBase::Pointer& generator)
{
  mitk::PixelBasedParameterFitImageGenerator::Pointer fitGenerator =
    mitk::PixelBasedParameterFitImageGenerator::New();

  mitk::DescriptivePharmacokineticBrixModelParameterizer::Pointer modelParameterizer =
    mitk::DescriptivePharmacokineticBrixModelParameterizer::New();

  //Model configuration (static parameters) can be done now
  modelParameterizer->SetTau(m_Controls.injectiontime->value());

  mitk::ImageTimeSelector::Pointer imageTimeSelector =	mitk::ImageTimeSelector::New();
  imageTimeSelector->SetInput(this->m_selectedImage);
  imageTimeSelector->SetTimeNr(0);
  imageTimeSelector->UpdateLargestPossibleRegion();

  mitk::DescriptivePharmacokineticBrixModelParameterizer::BaseImageType::Pointer baseImage;
  mitk::CastToItkImage(imageTimeSelector->GetOutput(), baseImage);

  modelParameterizer->SetBaseImage(baseImage);
  this->ConfigureInitialParametersOfParameterizer(modelParameterizer);

  //Specify fitting strategy and criterion parameters
  mitk::ModelFitFunctorBase::Pointer fitFunctor = CreateDefaultFitFunctor(modelParameterizer);

  //Parametrize fit generator
  fitGenerator->SetModelParameterizer(modelParameterizer);
  std::string roiUID = "";

  if (m_selectedMask.IsNotNull())
  {
    fitGenerator->SetMask(m_selectedMask);
    roiUID = m_selectedMask->GetUID();
  }

  fitGenerator->SetDynamicImage(m_selectedImage);
  fitGenerator->SetFitFunctor(fitFunctor);

  generator = fitGenerator.GetPointer();

  //Create model info
  modelFitInfo = mitk::modelFit::CreateFitInfoFromModelParameterizer(modelParameterizer,
    m_selectedNode->GetData(), mitk::ModelFitConstants::FIT_TYPE_VALUE_PIXELBASED(), this->GetFitName(), roiUID);
}

void MRPerfusionView::GenerateDescriptiveBrixModel_ROIBased(mitk::modelFit::ModelFitInfo::Pointer&
    modelFitInfo, mitk::ParameterFitImageGeneratorBase::Pointer& generator)
{
  if (m_selectedMask.IsNull())
  {
    return;
  }

  mitk::ROIBasedParameterFitImageGenerator::Pointer fitGenerator =
    mitk::ROIBasedParameterFitImageGenerator::New();

  mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::Pointer modelParameterizer =
    mitk::DescriptivePharmacokineticBrixModelValueBasedParameterizer::New();

  //Compute ROI signal
  mitk::MaskedDynamicImageStatisticsGenerator::Pointer signalGenerator =
    mitk::MaskedDynamicImageStatisticsGenerator::New();
  signalGenerator->SetMask(m_selectedMask);
  signalGenerator->SetDynamicImage(m_selectedImage);
  signalGenerator->Generate();

  mitk::MaskedDynamicImageStatisticsGenerator::ResultType roiSignal = signalGenerator->GetMean();

  //Model configuration (static parameters) can be done now
  modelParameterizer->SetTau(m_Controls.injectiontime->value());
  modelParameterizer->SetBaseValue(roiSignal[0]);
  this->ConfigureInitialParametersOfParameterizer(modelParameterizer);

  //Specify fitting strategy and criterion parameters
  mitk::ModelFitFunctorBase::Pointer fitFunctor = CreateDefaultFitFunctor(modelParameterizer);

  //Parametrize fit generator
  fitGenerator->SetModelParameterizer(modelParameterizer);
  fitGenerator->SetMask(m_selectedMask);
  fitGenerator->SetFitFunctor(fitFunctor);
  fitGenerator->SetSignal(roiSignal);
  fitGenerator->SetTimeGrid(mitk::ExtractTimeGrid(m_selectedImage));

  generator = fitGenerator.GetPointer();

  std::string roiUID = this->m_selectedMask->GetUID();

  //Create model info
  modelFitInfo = mitk::modelFit::CreateFitInfoFromModelParameterizer(modelParameterizer,
    m_selectedNode->GetData(), mitk::ModelFitConstants::FIT_TYPE_VALUE_ROIBASED(), this->GetFitName(), roiUID);
  mitk::ScalarListLookupTable::ValueType infoSignal;

  for (mitk::MaskedDynamicImageStatisticsGenerator::ResultType::const_iterator pos =
         roiSignal.begin(); pos != roiSignal.end(); ++pos)
  {
    infoSignal.push_back(*pos);
  }

  modelFitInfo->inputData.SetTableValue("ROI", infoSignal);
}

template <typename TParameterizer>
void MRPerfusionView::GenerateLinearModelFit_PixelBased(mitk::modelFit::ModelFitInfo::Pointer&
    modelFitInfo, mitk::ParameterFitImageGeneratorBase::Pointer& generator)
{
  mitk::PixelBasedParameterFitImageGenerator::Pointer fitGenerator =
    mitk::PixelBasedParameterFitImageGenerator::New();

  typename TParameterizer::Pointer modelParameterizer = TParameterizer::New();

  this->ConfigureInitialParametersOfParameterizer(modelParameterizer);

  //Specify fitting strategy and criterion parameters
  mitk::ModelFitFunctorBase::Pointer fitFunctor = CreateDefaultFitFunctor(modelParameterizer);

  //Parametrize fit generator
  fitGenerator->SetModelParameterizer(modelParameterizer);
  std::string roiUID = "";

  if (m_selectedMask.IsNotNull())
  {
    fitGenerator->SetMask(m_selectedMask);
    roiUID = this->m_selectedMask->GetUID();
  }

  fitGenerator->SetDynamicImage(m_selectedImage);
  fitGenerator->SetFitFunctor(fitFunctor);

  generator = fitGenerator.GetPointer();

  //Create model info
  modelFitInfo = mitk::modelFit::CreateFitInfoFromModelParameterizer(modelParameterizer,
    m_selectedNode->GetData(), mitk::ModelFitConstants::FIT_TYPE_VALUE_PIXELBASED(), this->GetFitName(), roiUID);
}

template <typename TParameterizer>
void MRPerfusionView::GenerateLinearModelFit_ROIBased(mitk::modelFit::ModelFitInfo::Pointer&
    modelFitInfo, mitk::ParameterFitImageGeneratorBase::Pointer& generator)
{
  if (m_selectedMask.IsNull())
  {
    return;
  }

  mitk::ROIBasedParameterFitImageGenerator::Pointer fitGenerator =
    mitk::ROIBasedParameterFitImageGenerator::New();

  typename TParameterizer::Pointer modelParameterizer = TParameterizer::New();

  //Compute ROI signal
  mitk::MaskedDynamicImageStatisticsGenerator::Pointer signalGenerator =
    mitk::MaskedDynamicImageStatisticsGenerator::New();
  signalGenerator->SetMask(m_selectedMask);
  signalGenerator->SetDynamicImage(m_selectedImage);
  signalGenerator->Generate();

  mitk::MaskedDynamicImageStatisticsGenerator::ResultType roiSignal = signalGenerator->GetMean();

  //Model configuration (static parameters) can be done now
  this->ConfigureInitialParametersOfParameterizer(modelParameterizer);

  //Specify fitting strategy and criterion parameters
  mitk::ModelFitFunctorBase::Pointer fitFunctor = CreateDefaultFitFunctor(modelParameterizer);

  //Parametrize fit generator
  fitGenerator->SetModelParameterizer(modelParameterizer);
  fitGenerator->SetMask(m_selectedMask);
  fitGenerator->SetFitFunctor(fitFunctor);
  fitGenerator->SetSignal(roiSignal);
  fitGenerator->SetTimeGrid(mitk::ExtractTimeGrid(m_selectedImage));

  generator = fitGenerator.GetPointer();

  std::string roiUID = this->m_selectedMask->GetUID();

  //Create model info
  modelFitInfo = mitk::modelFit::CreateFitInfoFromModelParameterizer(modelParameterizer,
    m_selectedNode->GetData(), mitk::ModelFitConstants::FIT_TYPE_VALUE_ROIBASED(), this->GetFitName(), roiUID);
  mitk::ScalarListLookupTable::ValueType infoSignal;

  for (mitk::MaskedDynamicImageStatisticsGenerator::ResultType::const_iterator pos =
         roiSignal.begin(); pos != roiSignal.end(); ++pos)
  {
    infoSignal.push_back(*pos);
  }

  modelFitInfo->inputData.SetTableValue("ROI", infoSignal);
}

template <typename TParameterizer>
void MRPerfusionView::GenerateAIFbasedModelFit_PixelBased(mitk::modelFit::ModelFitInfo::Pointer&
    modelFitInfo, mitk::ParameterFitImageGeneratorBase::Pointer& generator)
{
  mitk::PixelBasedParameterFitImageGenerator::Pointer fitGenerator =
    mitk::PixelBasedParameterFitImageGenerator::New();

  typename TParameterizer::Pointer modelParameterizer =
    TParameterizer::New();

  PrepareConcentrationImage();

  mitk::AIFBasedModelBase::AterialInputFunctionType aif;
  mitk::AIFBasedModelBase::AterialInputFunctionType aifTimeGrid;
  GetAIF(aif, aifTimeGrid);

  modelParameterizer->SetAIF(aif);
  modelParameterizer->SetAIFTimeGrid(aifTimeGrid);

  this->ConfigureInitialParametersOfParameterizer(modelParameterizer);


  //Specify fitting strategy and criterion parameters
  mitk::ModelFitFunctorBase::Pointer fitFunctor = CreateDefaultFitFunctor(modelParameterizer);

  //Parametrize fit generator
  fitGenerator->SetModelParameterizer(modelParameterizer);
  std::string roiUID = "";

  if (m_selectedMask.IsNotNull())
  {
    fitGenerator->SetMask(m_selectedMask);
    roiUID = this->m_selectedMask->GetUID();
  }

  fitGenerator->SetDynamicImage(this->m_inputImage);
  fitGenerator->SetFitFunctor(fitFunctor);

  generator = fitGenerator.GetPointer();

  //Create model info
  modelFitInfo = mitk::modelFit::CreateFitInfoFromModelParameterizer(modelParameterizer,
    this->m_inputImage, mitk::ModelFitConstants::FIT_TYPE_VALUE_PIXELBASED(), this->GetFitName(),
                 roiUID);

  mitk::ScalarListLookupTable::ValueType infoSignal;

  for (mitk::AIFBasedModelBase::AterialInputFunctionType::const_iterator pos =
         aif.begin(); pos != aif.end(); ++pos)
  {
    infoSignal.push_back(*pos);
  }

  modelFitInfo->inputData.SetTableValue("AIF", infoSignal);
}

template <typename TParameterizer>
void MRPerfusionView::GenerateAIFbasedModelFit_ROIBased(
  mitk::modelFit::ModelFitInfo::Pointer& modelFitInfo,
  mitk::ParameterFitImageGeneratorBase::Pointer& generator)
{
  if (m_selectedMask.IsNull())
  {
    return;
  }

  mitk::ROIBasedParameterFitImageGenerator::Pointer fitGenerator =
    mitk::ROIBasedParameterFitImageGenerator::New();

  typename TParameterizer::Pointer modelParameterizer =
    TParameterizer::New();

  PrepareConcentrationImage();

  mitk::AIFBasedModelBase::AterialInputFunctionType aif;
  mitk::AIFBasedModelBase::AterialInputFunctionType aifTimeGrid;
  GetAIF(aif, aifTimeGrid);

  modelParameterizer->SetAIF(aif);
  modelParameterizer->SetAIFTimeGrid(aifTimeGrid);

  this->ConfigureInitialParametersOfParameterizer(modelParameterizer);


  //Compute ROI signal
  mitk::MaskedDynamicImageStatisticsGenerator::Pointer signalGenerator =
    mitk::MaskedDynamicImageStatisticsGenerator::New();
  signalGenerator->SetMask(m_selectedMask);
  signalGenerator->SetDynamicImage(this->m_inputImage);
  signalGenerator->Generate();

  mitk::MaskedDynamicImageStatisticsGenerator::ResultType roiSignal = signalGenerator->GetMean();

  //Specify fitting strategy and criterion parameters
  mitk::ModelFitFunctorBase::Pointer fitFunctor = CreateDefaultFitFunctor(modelParameterizer);

  //Parametrize fit generator
  fitGenerator->SetModelParameterizer(modelParameterizer);
  fitGenerator->SetMask(m_selectedMask);
  fitGenerator->SetFitFunctor(fitFunctor);
  fitGenerator->SetSignal(roiSignal);
  fitGenerator->SetTimeGrid(mitk::ExtractTimeGrid(this->m_inputImage));

  generator = fitGenerator.GetPointer();

  std::string roiUID = this->m_selectedMask->GetUID();

  //Create model info
  modelFitInfo = mitk::modelFit::CreateFitInfoFromModelParameterizer(modelParameterizer,
    this->m_inputImage, mitk::ModelFitConstants::FIT_TYPE_VALUE_ROIBASED(), this->GetFitName(),
                 roiUID);

  mitk::ScalarListLookupTable::ValueType infoSignal;

  for (mitk::MaskedDynamicImageStatisticsGenerator::ResultType::const_iterator pos =
         roiSignal.begin(); pos != roiSignal.end(); ++pos)
  {
    infoSignal.push_back(*pos);
  }

  modelFitInfo->inputData.SetTableValue("ROI", infoSignal);

  infoSignal.clear();

  for (mitk::AIFBasedModelBase::AterialInputFunctionType::const_iterator pos =
         aif.begin(); pos != aif.end(); ++pos)
  {
    infoSignal.push_back(*pos);
  }

  modelFitInfo->inputData.SetTableValue("AIF", infoSignal);
}


void MRPerfusionView::DoFit(const mitk::modelFit::ModelFitInfo* fitSession,
                            mitk::ParameterFitImageGeneratorBase* generator)
{
  this->m_Controls.infoBox->append(QString("<font color='green'>" + QString("Fitting Data Set . . .") + QString ("</font>")));


  /////////////////////////
  //create job and put it into the thread pool
  mitk::modelFit::ModelFitResultNodeVectorType additionalNodes;
  if (m_HasGeneratedNewInput)
  {
    additionalNodes.push_back(m_inputNode);
  }
  if (m_HasGeneratedNewInputAIF)
  {
    additionalNodes.push_back(m_inputAIFNode);
  }

  ParameterFitBackgroundJob* pJob = new ParameterFitBackgroundJob(generator, fitSession,
      this->m_selectedNode, additionalNodes);
  pJob->setAutoDelete(true);

  connect(pJob, SIGNAL(Error(QString)), this, SLOT(OnJobError(QString)));
  connect(pJob, SIGNAL(Finished()), this, SLOT(OnJobFinished()));
  connect(pJob, SIGNAL(ResultsAreAvailable(mitk::modelFit::ModelFitResultNodeVectorType,
                       const ParameterFitBackgroundJob*)), this,
          SLOT(OnJobResultsAreAvailable(mitk::modelFit::ModelFitResultNodeVectorType,
                                        const ParameterFitBackgroundJob*)), Qt::BlockingQueuedConnection);

  connect(pJob, SIGNAL(JobProgress(double)), this, SLOT(OnJobProgress(double)));
  connect(pJob, SIGNAL(JobStatusChanged(QString)), this, SLOT(OnJobStatusChanged(QString)));

  QThreadPool* threadPool = QThreadPool::globalInstance();
  threadPool->start(pJob);
}

MRPerfusionView::MRPerfusionView() : m_FittingInProgress(false), m_HasGeneratedNewInput(false), m_HasGeneratedNewInputAIF(false)
{
  m_selectedImage = nullptr;
  m_selectedMask = nullptr;

  mitk::ModelFactoryBase::Pointer factory =
    mitk::DescriptivePharmacokineticBrixModelFactory::New().GetPointer();
  m_FactoryStack.push_back(factory);
  factory = mitk::StandardToftsModelFactory::New().GetPointer();
  m_FactoryStack.push_back(factory);
  factory = mitk::ExtendedToftsModelFactory::New().GetPointer();
  m_FactoryStack.push_back(factory);
  factory = mitk::TwoCompartmentExchangeModelFactory::New().GetPointer();
  m_FactoryStack.push_back(factory);

  mitk::NodePredicateDataType::Pointer isLabelSet = mitk::NodePredicateDataType::New("LabelSetImage");
  mitk::NodePredicateDataType::Pointer isImage = mitk::NodePredicateDataType::New("Image");
  mitk::NodePredicateProperty::Pointer isBinary = mitk::NodePredicateProperty::New("binary", mitk::BoolProperty::New(true));
  mitk::NodePredicateAnd::Pointer isLegacyMask = mitk::NodePredicateAnd::New(isImage, isBinary);
  mitk::NodePredicateDimension::Pointer is3D = mitk::NodePredicateDimension::New(3);
  mitk::NodePredicateOr::Pointer isMask = mitk::NodePredicateOr::New(isLegacyMask, isLabelSet);
  mitk::NodePredicateAnd::Pointer isNoMask = mitk::NodePredicateAnd::New(isImage, mitk::NodePredicateNot::New(isMask));
  mitk::NodePredicateAnd::Pointer is3DImage = mitk::NodePredicateAnd::New(isImage, is3D, isNoMask);

  this->m_IsMaskPredicate = mitk::NodePredicateAnd::New(isMask, mitk::NodePredicateNot::New(mitk::NodePredicateProperty::New("helper object"))).GetPointer();

  this->m_IsNoMaskImagePredicate = mitk::NodePredicateAnd::New(isNoMask, mitk::NodePredicateNot::New(mitk::NodePredicateProperty::New("helper object"))).GetPointer();

  auto isDynamicData = mitk::NodePredicateFunction::New([](const mitk::DataNode* node)
  {
    return  (node && node->GetData() && node->GetData()->GetTimeSteps() > 1);
  });

  auto modelFitResultRelationRule = mitk::ModelFitResultRelationRule::New();
  auto isNoModelFitNodePredicate = mitk::NodePredicateNot::New(modelFitResultRelationRule->GetConnectedSourcesDetector());

  this->m_isValidPDWImagePredicate = mitk::NodePredicateAnd::New(is3DImage, isNoModelFitNodePredicate);
  this->m_isValidTimeSeriesImagePredicate = mitk::NodePredicateAnd::New(isDynamicData, isImage, isNoMask);
}

void MRPerfusionView::OnJobFinished()
{
  this->m_Controls.infoBox->append(QString("Fitting finished."));
  this->m_FittingInProgress = false;
  this->UpdateGUIControls();
};

void MRPerfusionView::OnJobError(QString err)
{
  MITK_ERROR << err.toStdString().c_str();

  m_Controls.infoBox->append(QString("<font color='red'><b>") + err + QString("</b></font>"));
};

void MRPerfusionView::OnJobResultsAreAvailable(mitk::modelFit::ModelFitResultNodeVectorType results,
    const ParameterFitBackgroundJob* pJob)
{
  //Store the resulting parameter fit image via convenience helper function in data storage
  //(handles the correct generation of the nodes and their properties)

  mitk::modelFit::StoreResultsInDataStorage(this->GetDataStorage(), results, pJob->GetParentNode());
  //this stores the concentration image and AIF concentration image, if generated for this fit in the storage.
  //if not generated for this fit, relevant nodes are empty.
  mitk::modelFit::StoreResultsInDataStorage(this->GetDataStorage(), pJob->GetAdditionalRelevantNodes(), pJob->GetParentNode());



};

void MRPerfusionView::OnJobProgress(double progress)
{
  QString report = QString("Progress. ") + QString::number(progress);
  this->m_Controls.infoBox->append(report);
};

void MRPerfusionView::OnJobStatusChanged(QString info)
{
  this->m_Controls.infoBox->append(info);
}


void MRPerfusionView::InitModelComboBox() const
{
  this->m_Controls.comboModel->clear();
  this->m_Controls.comboModel->addItem(tr("No model selected"));

  for (ModelFactoryStackType::const_iterator pos = m_FactoryStack.begin();
       pos != m_FactoryStack.end(); ++pos)
  {
    this->m_Controls.comboModel->addItem(QString::fromStdString((*pos)->GetClassID()));
  }

  this->m_Controls.comboModel->setCurrentIndex(0);
};

mitk::DataNode::Pointer MRPerfusionView::GenerateConcentrationNode(mitk::Image* image,
    const std::string& nodeName) const
{
  if (!image)
  {
    mitkThrow() << "Cannot generate concentration node. Passed image is null. parameter name: ";
  }

  mitk::DataNode::Pointer result = mitk::DataNode::New();

  result->SetData(image);
  result->SetName(nodeName);
  result->SetVisibility(true);

  return result;
};


mitk::Image::Pointer MRPerfusionView::ConvertConcentrationImage(bool AIFMode)
{
  //Compute Concentration image
  mitk::ConcentrationCurveGenerator::Pointer concentrationGen =
    mitk::ConcentrationCurveGenerator::New();

  if (m_Controls.checkDedicatedAIFImage->isChecked() && AIFMode)
  {
    concentrationGen->SetDynamicImage(this->m_selectedAIFImage);
  }
  else
  {
    concentrationGen->SetDynamicImage(this->m_selectedImage);
  }

  concentrationGen->SetAbsoluteSignalEnhancement(m_Controls.radioButton_absoluteEnhancement->isChecked());
  concentrationGen->SetRelativeSignalEnhancement(m_Controls.radioButton_relativeEnchancement->isChecked());
  concentrationGen->SetUsingT1Map(m_Controls.radioButtonUsingT1viaVFA->isChecked());


  if (this->m_Controls.radioButtonUsingT1viaVFA->isChecked())
  {
      concentrationGen->SetRepetitionTime(m_Controls.TRSpinBox->value());
      concentrationGen->SetRelaxivity(m_Controls.RelaxivitySpinBox->value());
      concentrationGen->SetPDWImage(dynamic_cast<mitk::Image*>(m_Controls.PDWImageNodeSelector->GetSelectedNode()->GetData()));
      concentrationGen->SetBaselineStartTimeStep(m_Controls.spinBox_baselineStartTimeStep->value());
      concentrationGen->SetBaselineEndTimeStep(m_Controls.spinBox_baselineEndTimeStep->value());
      //Convert Flipangle from degree to radiant
      double alpha = m_Controls.FlipangleSpinBox->value()/360*2* boost::math::constants::pi<double>();
      concentrationGen->SetFlipAngle(alpha);
      double alphaPDW = m_Controls.FlipanglePDWSpinBox->value() / 360 * 2 * boost::math::constants::pi<double>();
      concentrationGen->SetFlipAnglePDW(alphaPDW);

  }
  else
  {
    concentrationGen->SetFactor(m_Controls.factorSpinBox->value());
    concentrationGen->SetBaselineStartTimeStep(m_Controls.spinBox_baselineStartTimeStep->value());
    concentrationGen->SetBaselineEndTimeStep(m_Controls.spinBox_baselineEndTimeStep->value());
  }


  mitk::Image::Pointer concentrationImage = concentrationGen->GetConvertedImage();

  return concentrationImage;
}

void MRPerfusionView::GetAIF(mitk::AIFBasedModelBase::AterialInputFunctionType& aif,
                             mitk::AIFBasedModelBase::AterialInputFunctionType& aifTimeGrid)
{
  if (this->m_Controls.radioAIFFile->isChecked())
  {
    aif.clear();
    aifTimeGrid.clear();

    aif.SetSize(AIFinputFunction.size());
    aifTimeGrid.SetSize(AIFinputGrid.size());

    aif.fill(0.0);
    aifTimeGrid.fill(0.0);

    itk::Array<double>::iterator aifPos = aif.begin();

    for (std::vector<double>::const_iterator pos = AIFinputFunction.begin();
         pos != AIFinputFunction.end(); ++pos, ++aifPos)
    {
      *aifPos = *pos;
    }

    itk::Array<double>::iterator gridPos = aifTimeGrid.begin();

    for (std::vector<double>::const_iterator pos = AIFinputGrid.begin(); pos != AIFinputGrid.end();
         ++pos, ++gridPos)
    {
      *gridPos = *pos;
    }
  }
  else if (this->m_Controls.radioAIFImage->isChecked())
  {
    aif.clear();
    aifTimeGrid.clear();

    mitk::AterialInputFunctionGenerator::Pointer aifGenerator =
      mitk::AterialInputFunctionGenerator::New();

    //Hematocrit level
    aifGenerator->SetHCL(this->m_Controls.HCLSpinBox->value());

    //mask settings
    this->m_selectedAIFMaskNode = m_Controls.AIFMaskNodeSelector->GetSelectedNode();
    this->m_selectedAIFMask = dynamic_cast<mitk::Image*>(this->m_selectedAIFMaskNode->GetData());

    if (this->m_selectedAIFMask->GetTimeSteps() > 1)
    {
      MITK_INFO <<
                "Selected AIF mask has multiple timesteps. Only use first timestep to mask model fit. AIF Mask name: "
                <<
                m_selectedAIFMaskNode->GetName() ;
      mitk::ImageTimeSelector::Pointer maskedImageTimeSelector = mitk::ImageTimeSelector::New();
      maskedImageTimeSelector->SetInput(this->m_selectedAIFMask);
      maskedImageTimeSelector->SetTimeNr(0);
      maskedImageTimeSelector->UpdateLargestPossibleRegion();
      this->m_selectedAIFMask = maskedImageTimeSelector->GetOutput();
    }

    if (this->m_selectedAIFMask.IsNotNull())
    {
      aifGenerator->SetMask(this->m_selectedAIFMask);
    }

    //image settings
    if (this->m_Controls.checkDedicatedAIFImage->isChecked())
    {
      this->m_selectedAIFImageNode = m_Controls.AIFImageNodeSelector->GetSelectedNode();
      this->m_selectedAIFImage = dynamic_cast<mitk::Image*>(this->m_selectedAIFImageNode->GetData());
    }
    else
    {
      this->m_selectedAIFImageNode = m_selectedNode;
      this->m_selectedAIFImage = m_selectedImage;
    }

    this->PrepareAIFConcentrationImage();

    aifGenerator->SetDynamicImage(this->m_inputAIFImage);

    aif = aifGenerator->GetAterialInputFunction();
    aifTimeGrid = aifGenerator->GetAterialInputFunctionTimeGrid();
  }
  else
  {
    mitkThrow() << "Cannot generate AIF. View is in a invalid state. No AIF mode selected.";
  }
}


void MRPerfusionView::LoadAIFfromFile()
{
  QFileDialog dialog;
  dialog.setNameFilter(tr("Images (*.csv"));

  QString fileName = dialog.getOpenFileName();

  m_Controls.aifFilePath->setText(fileName);

  std::string m_aifFilePath = fileName.toStdString();
  //Read Input
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  /////////////////////////////////////////////////////////////////////////////////////////////////
  //AIF Data

  std::ifstream in1(m_aifFilePath.c_str());

  if (!in1.is_open())
  {
    this->m_Controls.infoBox->append(QString("Could not open AIF File!"));
  }


  std::vector< std::string > vec1;
  std::string line1;

  while (getline(in1, line1))
  {
    Tokenizer tok(line1);
    vec1.assign(tok.begin(), tok.end());

    this->AIFinputGrid.push_back(convertToDouble(vec1[0]));
    this->AIFinputFunction.push_back(convertToDouble(vec1[1]));
  }
}

void MRPerfusionView::PrepareConcentrationImage()
{
  mitk::Image::Pointer concentrationImage = this->m_selectedImage;
  mitk::DataNode::Pointer concentrationNode = this->m_selectedNode;
  m_HasGeneratedNewInput = false;

  if (!this->m_Controls.radioButtonNoConversion->isChecked())
  {
    concentrationImage = this->ConvertConcentrationImage(false);
    concentrationNode = GenerateConcentrationNode(concentrationImage, "Concentration");
    m_HasGeneratedNewInput = true;
  }

  m_inputImage = concentrationImage;
  m_inputNode = concentrationNode;
}

void MRPerfusionView::PrepareAIFConcentrationImage()
{
  mitk::Image::Pointer concentrationImage = this->m_selectedImage;
  mitk::DataNode::Pointer concentrationNode = this->m_selectedNode;
  m_HasGeneratedNewInputAIF = false;

  if (this->m_Controls.checkDedicatedAIFImage->isChecked())
  {
    concentrationImage = this->m_selectedAIFImage;
    concentrationNode = this->m_selectedAIFImageNode;
  }

  if (!this->m_Controls.radioButtonNoConversion->isChecked())
  {
    if (!this->m_Controls.checkDedicatedAIFImage->isChecked())
    {
      if (m_inputImage.IsNull())
      {
        mitkThrow() <<
          "Cannot get AIF concentration image. Invalid view state. Input image is not defined yet, but should be.";
      }

      //we can directly use the concentration input image/node (generated by GetConcentrationImage) also for the AIF
      concentrationImage = this->m_inputImage;
      concentrationNode = this->m_inputNode;
    }
    else
    {
      concentrationImage = this->ConvertConcentrationImage(true);
      concentrationNode = GenerateConcentrationNode(concentrationImage, "AIF Concentration");
      m_HasGeneratedNewInputAIF = true;
    }
  }

  m_inputAIFImage = concentrationImage;
  m_inputAIFNode = concentrationNode;
}



mitk::ModelFitFunctorBase::Pointer MRPerfusionView::CreateDefaultFitFunctor(
  const mitk::ModelParameterizerBase* parameterizer) const
{
  mitk::LevenbergMarquardtModelFitFunctor::Pointer fitFunctor =
    mitk::LevenbergMarquardtModelFitFunctor::New();

  mitk::NormalizedSumOfSquaredDifferencesFitCostFunction::Pointer chi2 =
    mitk::NormalizedSumOfSquaredDifferencesFitCostFunction::New();
  fitFunctor->RegisterEvaluationParameter("Chi^2", chi2);

  if (m_Controls.checkBox_Constraints->isChecked())
  {
    fitFunctor->SetConstraintChecker(m_modelConstraints);
  }

  mitk::ModelBase::Pointer refModel = parameterizer->GenerateParameterizedModel();

  ::itk::LevenbergMarquardtOptimizer::ScalesType scales;
  scales.SetSize(refModel->GetNumberOfParameters());
  scales.Fill(1.0);
  fitFunctor->SetScales(scales);

  fitFunctor->SetDebugParameterMaps(m_Controls.checkDebug->isChecked());

  return fitFunctor.GetPointer();
}
