/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#include "QmitkStoreSCPLauncher.h"
#include <QMessageBox>
#include <QProcessEnvironment>
#include <mitkLog.h>

#include <fstream>
#include <iostream>
#include <QFile>
#include <QTextStream>
#include <QIODevice>
#include <QDir>
#include <QDirIterator>
#include <QCoreApplication>
#include "org_mitk_gui_qt_dicombrowser_config.h"

QmitkStoreSCPLauncher::QmitkStoreSCPLauncher(QmitkStoreSCPLauncherBuilder* builder)
: m_StoreSCP(new QProcess())
{
  m_StoreSCP->setProcessChannelMode(QProcess::MergedChannels);
    connect( m_StoreSCP, SIGNAL(error(QProcess::ProcessError)),this, SLOT(OnProcessError(QProcess::ProcessError)));
    connect( m_StoreSCP, SIGNAL(stateChanged(QProcess::ProcessState)),this, SLOT(OnStateChanged(QProcess::ProcessState)));
    connect( m_StoreSCP, SIGNAL(readyReadStandardOutput()),this, SLOT(OnReadyProcessOutput()));
    SetArgumentList(builder);
}

QmitkStoreSCPLauncher::~QmitkStoreSCPLauncher()
{
    disconnect( m_StoreSCP, SIGNAL(error(QProcess::ProcessError)),this, SLOT(OnProcessError(QProcess::ProcessError)));
    disconnect( m_StoreSCP, SIGNAL(stateChanged(QProcess::ProcessState)),this, SLOT(OnStateChanged(QProcess::ProcessState)));
    disconnect( m_StoreSCP, SIGNAL(readyReadStandardOutput()),this, SLOT(OnReadyProcessOutput()));
    m_StoreSCP->close();
    m_StoreSCP->waitForFinished(1000);
    delete m_StoreSCP;
}

void QmitkStoreSCPLauncher::StartStoreSCP()
{
    FindPathToStoreSCP();
    m_StoreSCP->start(m_PathToStoreSCP,m_ArgumentList);
}

void QmitkStoreSCPLauncher::FindPathToStoreSCP()
{
    QString appPath= QCoreApplication::applicationDirPath();
    if(m_PathToStoreSCP.isEmpty())
    {
        QString fileName;
#ifdef _WIN32
        fileName = "/storescp.exe";
#else
        fileName = "/storescp";
#endif

        m_PathToStoreSCP = appPath + fileName;

        //In development the storescp isn't copied into bin directory
        if(!QFile::exists(m_PathToStoreSCP))
        {
            m_PathToStoreSCP = static_cast<QString>(DCMTK_STORESCP);
        }
    }
}

void QmitkStoreSCPLauncher::OnReadyProcessOutput()
{
    QString out(m_StoreSCP->readAllStandardOutput());
    QStringList allDataList,importList;

    allDataList = out.split("\n",Qt::SkipEmptyParts);
    QStringListIterator it(allDataList);

    while(it.hasNext())
    {
        QString output = it.next();
        if (output.contains("E: "))
        {
            output.replace("E: ","");
            m_ErrorText = output;
            OnProcessError(QProcess::UnknownError);
            return;
        }
        if(output.contains("I: storing DICOM file: "))
        {
            output.replace("I: storing DICOM file: ","");
            output.replace("\\", "/"); // cannot handle backslashes
            output.replace("\r", ""); // cannot handle carriage return
            importList += output;
        }
    }
    if(!importList.isEmpty())
    {
        emit SignalStartImport(importList);
    }
}

void QmitkStoreSCPLauncher::OnProcessError(QProcess::ProcessError err)
{
    switch(err)
    {
    case QProcess::FailedToStart:
        m_ErrorText.prepend("Failed to start storage provider: ");
        m_ErrorText.append(m_StoreSCP->errorString());
        emit SignalStoreSCPError(m_ErrorText);
        m_ErrorText.clear();
        break;
    case QProcess::Crashed:
        m_ErrorText.prepend("Storage provider closed: ");
        m_ErrorText.append(m_StoreSCP->errorString());
        emit SignalStoreSCPError(m_ErrorText);
        m_ErrorText.clear();
        break;
    case QProcess::Timedout:
        m_ErrorText.prepend("Storage provider timeout: ");
        m_ErrorText.append(m_StoreSCP->errorString());
        emit SignalStoreSCPError(m_ErrorText);
        m_ErrorText.clear();
        break;
    case QProcess::WriteError:
        m_ErrorText.prepend("Storage provider write error: ");
        m_ErrorText.append(m_StoreSCP->errorString());
        emit SignalStoreSCPError(m_ErrorText);
        m_ErrorText.clear();
        break;
    case QProcess::ReadError:
        m_ErrorText.prepend("Storage provider read error: ");
        m_ErrorText.append(m_StoreSCP->errorString());
        emit SignalStoreSCPError(m_ErrorText);
        m_ErrorText.clear();
        break;
    case QProcess::UnknownError:
        m_ErrorText.prepend("Storage provider unknown error: ");
        m_ErrorText.append(m_StoreSCP->errorString());
        emit SignalStoreSCPError(m_ErrorText);
        m_ErrorText.clear();
        break;
    default:
        m_ErrorText.prepend("Storage provider unknown error: ");
        m_ErrorText.append(m_StoreSCP->errorString());
        emit SignalStoreSCPError(m_ErrorText);
        m_ErrorText.clear();
        break;
    }
}

void QmitkStoreSCPLauncher::OnStateChanged(QProcess::ProcessState status)
{
    switch(status)
    {
    case QProcess::NotRunning:
        m_StatusText.prepend("Storage provider not running!");
        emit SignalStatusOfStoreSCP(m_StatusText);
        m_StatusText.clear();
        break;
    case QProcess::Starting:
        m_StatusText.prepend("Starting storage provider!");
        emit SignalStatusOfStoreSCP(m_StatusText);
        m_StatusText.clear();
        break;
    case QProcess::Running:
        m_StatusText.prepend(m_ArgumentList[0]).prepend(" Port: ").prepend(m_ArgumentList[2]).prepend(" AET: ").prepend("Storage provider running! ");
        emit SignalStatusOfStoreSCP(m_StatusText);
        m_StatusText.clear();
        break;
    default:
        m_StatusText.prepend("Storage provider unknown error!");
        emit SignalStatusOfStoreSCP(m_StatusText);
        m_StatusText.clear();
        break;
    }
}

void QmitkStoreSCPLauncher::SetArgumentList(QmitkStoreSCPLauncherBuilder* builder)
{
    m_ArgumentList << *builder->GetPort() << QString("-aet") <<*builder->GetAETitle() << *builder->GetTransferSyntax()
        << *builder->GetOtherNetworkOptions() << *builder->GetMode() << QString("-od") << *builder->GetOutputDirectory();
}

QString QmitkStoreSCPLauncher::ArgumentListToQString()
{
    QString argumentString;
    QStringListIterator argumentIterator(m_ArgumentList);
    while(argumentIterator.hasNext())
    {
        argumentString.append(" ");
        argumentString.append(argumentIterator.next());
    }
    return argumentString;
}
