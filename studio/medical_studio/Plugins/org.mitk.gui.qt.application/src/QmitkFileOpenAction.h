/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkFileOpenAction_h
#define QmitkFileOpenAction_h

#include <org_mitk_gui_qt_application_Export.h>

#include <berryIWorkbenchWindow.h>

// qt
#include <QAction>
#include <QIcon>

class QmitkFileOpenActionPrivate;

class MITK_QT_APP QmitkFileOpenAction : public QAction
{
  Q_OBJECT

public:

  QmitkFileOpenAction(berry::IWorkbenchWindow::Pointer window);
  QmitkFileOpenAction(const QIcon& icon, berry::IWorkbenchWindow::Pointer window);
  QmitkFileOpenAction(const QIcon& icon, berry::IWorkbenchWindow* window);

  ~QmitkFileOpenAction() override;

protected slots:

  virtual void Run();

private:

  const QScopedPointer<QmitkFileOpenActionPrivate> d;

};


#endif
