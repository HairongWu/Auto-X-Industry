/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

// Qmitk
#include "QmitkDicomLocalStorageWidget.h"

// Qt
#include <QLabel>
#include <QMessageBox>
#include <QProgressDialog>
#include <QVariant>

#include <ctkDICOMIndexer.h>

const std::string QmitkDicomLocalStorageWidget::Widget_ID = "org.mitk.Widgets.QmitkDicomLocalStorageWidget";

QmitkDicomLocalStorageWidget::QmitkDicomLocalStorageWidget(QWidget *parent)
  : QWidget(parent), m_LocalIndexer(new ctkDICOMIndexer(parent)), m_Controls(nullptr)
{
  CreateQtPartControl(this);
}

QmitkDicomLocalStorageWidget::~QmitkDicomLocalStorageWidget()
{
  m_LocalDatabase->closeDatabase();
}

void QmitkDicomLocalStorageWidget::CreateQtPartControl(QWidget *parent)
{
  if (!m_Controls)
  {
    m_Controls = new Ui::QmitkDicomLocalStorageWidgetControls;
    m_Controls->setupUi(parent);

    connect(m_Controls->deleteButton, SIGNAL(clicked()), this, SLOT(OnDeleteButtonClicked()));
    connect(m_Controls->viewInternalDataButton, SIGNAL(clicked()), this, SLOT(OnViewButtonClicked()));

    connect(m_Controls->ctkDicomBrowser,
            SIGNAL(seriesSelectionChanged(const QStringList &)),
            this,
            SLOT(OnSeriesSelectionChanged(const QStringList &)));
    connect(m_Controls->ctkDicomBrowser,
            SIGNAL(seriesSelectionChanged(const QStringList &)),
            this,
            SLOT(OnSeriesSelectionChanged(const QStringList &)));
    connect(
      m_Controls->ctkDicomBrowser, SIGNAL(seriesDoubleClicked(const QModelIndex &)), this, SLOT(OnViewButtonClicked()));

    connect(m_LocalIndexer, SIGNAL(indexingComplete(int, int, int, int)), this, SIGNAL(SignalFinishedImport()));

    m_Controls->ctkDicomBrowser->setTableOrientation(Qt::Vertical);
  }
}

void QmitkDicomLocalStorageWidget::OnStartDicomImport(const QString &dicomData)
{
  if (m_LocalDatabase->isOpen())
  {
    m_LocalIndexer->addDirectory(dicomData);
  }
}

void QmitkDicomLocalStorageWidget::OnStartDicomImport(const QStringList &dicomData)
{
  if (m_LocalDatabase->isOpen())
  {
    m_LocalIndexer->addListOfFiles( dicomData);
  }
}

void QmitkDicomLocalStorageWidget::OnDeleteButtonClicked()
{
  if (!this->DeletePatients())
  {
    if (!this->DeleteStudies())
    {
      this->DeleteSeries();
    }
  }

  m_Controls->ctkDicomBrowser->updateTableViews();
}

bool QmitkDicomLocalStorageWidget::DeletePatients()
{
  auto selectedPatientUIDs = m_Controls->ctkDicomBrowser->currentPatientsSelection();

  if (!selectedPatientUIDs.empty())
  {
    QStringList studyUIDs;

    for (const auto &patientUID : std::as_const(selectedPatientUIDs))
      studyUIDs.append(m_LocalDatabase->studiesForPatient(patientUID));

    QStringList seriesUIDs;

    for (const auto &studyUID : studyUIDs)
      seriesUIDs.append(m_LocalDatabase->seriesForStudy(studyUID));

    auto answer = QMessageBox::question(nullptr,
                                        "Delete Patients",
                                        QString("Do you really want to delete %1 %2, containing %3 series in %4 %5?")
                                          .arg(selectedPatientUIDs.count())
                                          .arg(selectedPatientUIDs.count() != 1 ? "patients" : "patient")
                                          .arg(seriesUIDs.count())
                                          .arg(studyUIDs.count())
                                          .arg(studyUIDs.count() != 1 ? "studies" : "study"),
                                        QMessageBox::Yes | QMessageBox::No,
                                        QMessageBox::No);

    if (answer == QMessageBox::Yes)
    {
      for (const auto &patientUID : std::as_const(selectedPatientUIDs))
        m_LocalDatabase->removePatient(patientUID);
    }

    return true;
  }

  return false;
}

bool QmitkDicomLocalStorageWidget::DeleteStudies()
{
  auto selectedStudyUIDs = m_Controls->ctkDicomBrowser->currentStudiesSelection();

  if (!selectedStudyUIDs.empty())
  {
    QStringList seriesUIDs;

    for (const auto &studyUID : std::as_const(selectedStudyUIDs))
      seriesUIDs.append(m_LocalDatabase->seriesForStudy(studyUID));

    auto answer = QMessageBox::question(nullptr,
                                        "Delete Studies",
                                        QString("Do you really want to delete %1 %2, containing %3 series?")
                                          .arg(selectedStudyUIDs.count())
                                          .arg(selectedStudyUIDs.count() != 1 ? "studies" : "study")
                                          .arg(seriesUIDs.count()),
                                        QMessageBox::Yes | QMessageBox::No,
                                        QMessageBox::No);

    if (answer == QMessageBox::Yes)
    {
      for (const auto &studyUID : std::as_const(selectedStudyUIDs))
        m_LocalDatabase->removeStudy(studyUID);
    }

    return true;
  }

  return false;
}

bool QmitkDicomLocalStorageWidget::DeleteSeries()
{
  auto selectedSeriesUIDs = m_Controls->ctkDicomBrowser->currentSeriesSelection();

  if (!selectedSeriesUIDs.empty())
  {
    auto answer =
      QMessageBox::question(nullptr,
                            "Delete Series",
                            QString("Do you really want to delete %1 series?").arg(selectedSeriesUIDs.count()),
                            QMessageBox::Yes | QMessageBox::No,
                            QMessageBox::No);

    if (answer == QMessageBox::Yes)
    {
      for (const auto &seriesUID : std::as_const(selectedSeriesUIDs))
        m_LocalDatabase->removeSeries(seriesUID);
    }

    return true;
  }

  return false;
}

void QmitkDicomLocalStorageWidget::OnViewButtonClicked()
{
  QStringList uids = m_Controls->ctkDicomBrowser->currentSeriesSelection();
  QString uid;
  foreach (uid, uids)
  {
    QStringList filesForSeries = m_LocalDatabase->filesForSeries(uid);
    QHash<QString, QVariant> eventProperty;
    eventProperty.insert("FilesForSeries", filesForSeries);
    if (!filesForSeries.isEmpty())
    {
      QString modality = m_LocalDatabase->fileValue(filesForSeries.at(0), "0008,0060");
      eventProperty.insert("Modality", modality);
    }
    emit SignalDicomToDataManager(eventProperty);
  }
}

void QmitkDicomLocalStorageWidget::SetDatabaseDirectory(QString newDatatbaseDirectory)
{
  QDir databaseDirecory = QDir(newDatatbaseDirectory);
  if (!databaseDirecory.exists())
  {
    databaseDirecory.mkpath(databaseDirecory.absolutePath());
  }
  QString newDatatbaseFile = databaseDirecory.absolutePath() + QString("/ctkDICOM.sql");
  this->SetDatabase(newDatatbaseFile);
}

void QmitkDicomLocalStorageWidget::SetDatabase(QString databaseFile)
{
  m_LocalDatabase = new ctkDICOMDatabase(databaseFile);
  m_LocalDatabase->setParent(this);
  m_Controls->ctkDicomBrowser->setDICOMDatabase(m_LocalDatabase);
  m_LocalIndexer->setDatabase(m_LocalDatabase);
}

void QmitkDicomLocalStorageWidget::OnSeriesSelectionChanged(const QStringList &s)
{
  m_Controls->viewInternalDataButton->setEnabled((s.size() != 0));
}
