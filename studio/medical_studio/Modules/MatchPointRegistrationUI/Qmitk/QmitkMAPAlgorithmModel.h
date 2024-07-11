/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef QmitkMAPAlgorithmModel_h
#define QmitkMAPAlgorithmModel_h

#include <QAbstractTableModel>
#include <QStringList>

// MITK
#include "MitkMatchPointRegistrationUIExports.h"

// MatchPoint
#include <mapMetaPropertyAlgorithmInterface.h>
#include <mapRegistrationAlgorithmBase.h>

/*!
  \class QmitkMAPAlgorithmModel
  Helper class that implements a model to handle the MetaProperty interface of a MatchPoint algorithm
  in context of the QT view-model-concept. A algorithm can be set as data source for the model.
  The model retrieves all information through the MetaPropertyInterface. Changes in the view will
  be propagated by the model into the algorithm.
  \remarks The model only keep a simple pointer to the MetaPropertyInterface of the algorithm.
   You have to ensure to reset the algorithm if the pointer goes invalid.
  \warning  This class is not yet documented. Use "git blame" and ask the author to provide basic documentation.
*/
class MITKMATCHPOINTREGISTRATIONUI_EXPORT QmitkMAPAlgorithmModel : public QAbstractTableModel
{
  Q_OBJECT

public:
  QmitkMAPAlgorithmModel(QObject *parent = nullptr);
  ~QmitkMAPAlgorithmModel() override{};

  void SetAlgorithm(map::algorithm::RegistrationAlgorithmBase *pAlgorithm);
  void SetAlgorithm(map::algorithm::facet::MetaPropertyAlgorithmInterface *pMetaInterface);

  Qt::ItemFlags flags(const QModelIndex &index) const override;
  QVariant data(const QModelIndex &index, int role) const override;
  QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
  int rowCount(const QModelIndex &parent = QModelIndex()) const override;
  int columnCount(const QModelIndex &parent = QModelIndex()) const override;
  bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole) override;

private:
  void UpdateMetaProperties() const;

  /** Method uses m_pMetaInterface to retrieve the MetaProperty and unwraps it into an
   * suitable QVariant depending on the passed QT role. If the MetaProperty type is not supported, the QVariant is
   * invalid.
   */
  QVariant GetPropertyValue(const map::algorithm::MetaPropertyInfo *pInfo, int role) const;

  template <typename TValueType>
  bool CheckCastAndSetProp(const map::algorithm::MetaPropertyInfo *pInfo, const QVariant &value);

  bool SetPropertyValue(const map::algorithm::MetaPropertyInfo *pInfo, const QVariant &value);

  map::algorithm::facet::MetaPropertyAlgorithmInterface *m_pMetaInterface;
  mutable map::algorithm::facet::MetaPropertyAlgorithmInterface::MetaPropertyVectorType m_MetaProperties;
};

#endif
