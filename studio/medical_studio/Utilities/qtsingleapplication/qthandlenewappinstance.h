/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QTHANDLENEWAPPINSTANCE_H
#define QTHANDLENEWAPPINSTANCE_H

#include <QString>

class QtSingleApplication;

bool createTemporaryDir(QString &path);

QString handleNewAppInstance(QtSingleApplication *singleApp, int argc, char **argv, const QString &newInstanceArg);

#endif // QTHANDLENEWAPPINSTANCE_H
