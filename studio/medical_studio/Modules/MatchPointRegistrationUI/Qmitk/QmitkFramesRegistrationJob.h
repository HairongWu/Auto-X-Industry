/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkFramesRegistrationJob_h
#define QmitkFramesRegistrationJob_h

// QT
#include <QObject>
#include <QRunnable>

// ITK
#include <itkCommand.h>

// MITK
#include <QmitkMappingJob.h>
#include <mitkDataNode.h>
#include <mitkImage.h>

// MatchPoint
#include <mapDeploymentDLLInfo.h>
#include <mapIterativeAlgorithmInterface.h>
#include <mapMultiResRegistrationAlgorithmInterface.h>
#include <mapRegistrationAlgorithmBase.h>
#include <mapRegistrationBase.h>

// Map4CTK
#include "mitkUIDHelper.h"
#include <mitkTimeFramesRegistrationHelper.h>

#include <MitkMatchPointRegistrationUIExports.h>

/** Simple helper job class that could be used to process a frame registration in a parallel thread.
 * This is e.g. used be plugins to keep the GUI responsive while doing a frame registration*/
class MITKMATCHPOINTREGISTRATIONUI_EXPORT QmitkFramesRegistrationJob : public QObject,
                                                                       public QRunnable,
                                                                       public QmitkMappingJobSettings
{
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT

public:
  QmitkFramesRegistrationJob(map::algorithm::RegistrationAlgorithmBase *pAlgorithm);
  ~QmitkFramesRegistrationJob() override;

  void run() override;

signals:
  void Finished();
  void Error(QString err);
  void ResultIsAvailable(mitk::Image::Pointer spResult, const QmitkFramesRegistrationJob *pJob);
  void AlgorithmIterated(QString info, bool hasIterationCount, unsigned long currentIteration);
  void LevelChanged(QString info, bool hasLevelCount, unsigned long currentLevel);
  void AlgorithmStatusChanged(QString info);
  void AlgorithmInfo(QString info);
  void FrameProcessed(double progress);
  void FrameRegistered(double progress);
  void FrameMapped(double progress);

public:
  // Inputs
  mitk::BaseData::ConstPointer m_spTargetData;

  mitk::Image::ConstPointer m_spTargetMask;

  // job settings
  mitk::TimeFramesRegistrationHelper::IgnoreListType m_IgnoreList;
  mitk::NodeUIDType m_TargetDataUID;
  mitk::NodeUIDType m_TargetMaskDataUID;

  const map::algorithm::RegistrationAlgorithmBase *GetLoadedAlgorithm() const;

private:
  typedef map::algorithm::facet::IterativeAlgorithmInterface IIterativeAlgorithm;
  typedef map::algorithm::facet::MultiResRegistrationAlgorithmInterface IMultiResAlgorithm;

  mitk::Image::Pointer m_spMappedImageNode;

  ::itk::MemberCommand<QmitkFramesRegistrationJob>::Pointer m_spCommand;
  unsigned long m_ObserverID;
  map::algorithm::RegistrationAlgorithmBase::Pointer m_spLoadedAlgorithm;

  mitk::TimeFramesRegistrationHelper::Pointer m_helper;

  // Helper functions
  const mitk::Image *GetTargetDataAsImage() const;

  void OnMapAlgorithmEvent(::itk::Object *, const itk::EventObject &event);
};

#endif
