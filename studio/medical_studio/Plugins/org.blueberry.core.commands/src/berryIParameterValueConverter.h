/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef BERRYABSTRACTPARAMETERVALUECONVERTER_H_
#define BERRYABSTRACTPARAMETERVALUECONVERTER_H_

#include <berryObject.h>
#include <berryMacros.h>

#include <org_blueberry_core_commands_Export.h>

namespace berry {

/**
 * <p>
 * Supports conversion between objects and strings for command parameter values.
 * Extenders must produce strings that identify objects (of a specific command
 * parameter type) as well as consume the strings to locate and return the
 * objects they identify.
 * </p>
 * <p>
 * This class offers multiple handlers of a command a consistent way of
 * converting string parameter values into the objects that the handlers would
 * prefer to deal with. This class also gives clients a way to serialize
 * object parameters as strings so that entire parameterized commands can be
 * serialized, stored and later deserialized and executed.
 * </p>
 * <p>
 * This class will typically be extended so the subclass can be referenced from
 * the <code>converter</code> attribute of the
 * <code>commandParameterType</code> element of the
 * <code>org.blueberry.ui.commands</code> extension-point. Objects implementing
 * this interface may also be passed directly to
 * {@link ParameterType#Define} by clients.
 * </p>
 *
 * @see ParameterType#Define(IParameterValueConverter::Pointer)
 * @see ParameterizedCommand#Serialize()
 */
struct BERRY_COMMANDS IParameterValueConverter {

  virtual ~IParameterValueConverter();

  /**
   * Converts a string encoded command parameter value into the parameter
   * value object.
   *
   * @param parameterValue
   *            a command parameter value string describing an object; may be
   *            <code>null</code>
   * @return the object described by the command parameter value string; may
   *         be <code>null</code>
   * @throws ParameterValueConversionException
   *             if an object cannot be produced from the
   *             <code>parameterValue</code> string
   */
  virtual Object::Pointer ConvertToObject(const QString& parameterValue) = 0;

  /**
   * Converts a command parameter value object into a string that encodes a
   * reference to the object or serialization of the object.
   *
   * @param parameterValue
   *            an object to convert into an identifying string; may be
   *            <code>null</code>
   * @return a string describing the provided object; may be <code>null</code>
   * @throws ParameterValueConversionException
   *             if a string reference or serialization cannot be provided for
   *             the <code>parameterValue</code>
   */
  virtual QString ConvertToString(const Object::Pointer& parameterValue) = 0;

};

}

Q_DECLARE_INTERFACE(berry::IParameterValueConverter, "org.blueberry.core.commands.IParameterValueConverter")

#endif /* BERRYABSTRACTPARAMETERVALUECONVERTER_H_ */
