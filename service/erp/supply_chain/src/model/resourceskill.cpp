/***************************************************************************
 *                                                                         *
 * Copyright (C) 2007-2015 by frePPLe bv                                   *
 *                                                                         *
 * Permission is hereby granted, free of charge, to any person obtaining   *
 * a copy of this software and associated documentation files (the         *
 * "Software"), to deal in the Software without restriction, including     *
 * without limitation the rights to use, copy, modify, merge, publish,     *
 * distribute, sublicense, and/or sell copies of the Software, and to      *
 * permit persons to whom the Software is furnished to do so, subject to   *
 * the following conditions:                                               *
 *                                                                         *
 * The above copyright notice and this permission notice shall be          *
 * included in all copies or substantial portions of the Software.         *
 *                                                                         *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,         *
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF      *
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                   *
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE  *
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION  *
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION   *
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.         *
 *                                                                         *
 ***************************************************************************/

#define FREPPLE_CORE
#include "frepple/model.h"

namespace frepple {

const MetaCategory* ResourceSkill::metadata;
const MetaClass* ResourceSkillDefault::metadata;

int ResourceSkill::initialize() {
  // Initialize the metadata
  metadata = MetaCategory::registerCategory<ResourceSkill>(
      "resourceskill", "resourceskills",
      Association<Resource, Skill, ResourceSkill>::reader, finder);
  registerFields<ResourceSkill>(const_cast<MetaCategory*>(metadata));
  ResourceSkillDefault::metadata =
      MetaClass::registerClass<ResourceSkillDefault>(
          "resourceskill", "resourceskill",
          Object::create<ResourceSkillDefault>, true);

  // Initialize the Python class
  PythonType& x = FreppleCategory<ResourceSkill>::getPythonType();
  x.setName("resourceskill");
  x.setDoc("frePPLe resourceskill");
  x.supportgetattro();
  x.supportsetattro();
  x.supportcreate(create);
  x.addMethod("toXML", toXML, METH_VARARGS, "return a XML representation");
  metadata->setPythonClass(x);
  return x.typeReady();
}

ResourceSkill::ResourceSkill(Skill* s, Resource* r, int u) {
  setSkill(s);
  setResource(r);
  setPriority(u);
  initType(metadata);
}

ResourceSkill::ResourceSkill(Skill* s, Resource* r, int u, DateRange e) {
  setSkill(s);
  setResource(r);
  setPriority(u);
  setEffective(e);
  initType(metadata);
}

ResourceSkill::~ResourceSkill() {
  // Delete the associated from the related objects
  if (getResource()) getResource()->skills.erase(this);
  if (getSkill()) getSkill()->resources.erase(this);
}

/* @todo this method implementation is not generic enough and not extendible by
 * subclasses. */
PyObject* ResourceSkill::create(PyTypeObject* pytype, PyObject* args,
                                PyObject* kwds) {
  try {
    // Pick up the skill
    PyObject* skill = PyDict_GetItemString(kwds, "skill");
    if (!skill) throw DataException("Missing skill on ResourceSkill");
    if (!PyObject_TypeCheck(skill, Skill::metadata->pythonClass))
      throw DataException("resourceskill skill must be of type skill");

    // Pick up the resource
    PyObject* res = PyDict_GetItemString(kwds, "resource");
    if (!skill) throw DataException("Missing resource on ResourceSkill");
    if (!PyObject_TypeCheck(res, Resource::metadata->pythonClass))
      throw DataException("resourceskill resource must be of type resource");

    // Pick up the priority
    PyObject* q1 = PyDict_GetItemString(kwds, "priority");
    int q2 = q1 ? PythonData(q1).getInt() : 1;

    // Pick up the effective dates
    DateRange eff;
    PyObject* eff_start = PyDict_GetItemString(kwds, "effective_start");
    if (eff_start) {
      PythonData d(eff_start);
      eff.setStart(d.getDate());
    }
    PyObject* eff_end = PyDict_GetItemString(kwds, "effective_end");
    if (eff_end) {
      PythonData d(eff_end);
      eff.setEnd(d.getDate());
    }

    // Create the resourceskill
    ResourceSkill* l = new ResourceSkill(static_cast<Skill*>(skill),
                                         static_cast<Resource*>(res), q2, eff);

    // Iterate over extra keywords, and set attributes.   @todo move this
    // responsibility to the readers...
    if (l) {
      PyObject *key, *value;
      Py_ssize_t pos = 0;
      while (PyDict_Next(kwds, &pos, &key, &value)) {
        PythonData field(value);
        PyObject* key_utf8 = PyUnicode_AsUTF8String(key);
        DataKeyword attr(PyBytes_AsString(key_utf8));
        Py_DECREF(key_utf8);
        if (!attr.isA(Tags::effective_end) &&
            !attr.isA(Tags::effective_start) && !attr.isA(Tags::skill) &&
            !attr.isA(Tags::resource) && !attr.isA(Tags::priority) &&
            !attr.isA(Tags::type) && !attr.isA(Tags::action)) {
          const MetaFieldBase* fmeta = l->getType().findField(attr.getHash());
          if (!fmeta && l->getType().category)
            fmeta = l->getType().category->findField(attr.getHash());
          if (fmeta)
            // Update the attribute
            fmeta->setField(l, field);
          else
            l->setProperty(attr.getName(), value);
        }
      };
    }

    // Return the object
    Py_INCREF(l);
    return static_cast<PyObject*>(l);
  } catch (...) {
    PythonType::evalException();
    return nullptr;
  }
}

Object* ResourceSkill::finder(const DataValueDict& d) {
  // Check resource
  const DataValue* tmp = d.get(Tags::resource);
  if (!tmp) return nullptr;
  Resource* res = static_cast<Resource*>(tmp->getObject());

  // Check skill field
  tmp = d.get(Tags::skill);
  if (!tmp) return nullptr;
  Skill* skill = static_cast<Skill*>(tmp->getObject());

  // Walk over all skills of the resource, and return
  // the first one with matching
  const DataValue* hasEffectiveStart = d.get(Tags::effective_start);
  Date effective_start;
  if (hasEffectiveStart) effective_start = hasEffectiveStart->getDate();
  const DataValue* hasEffectiveEnd = d.get(Tags::effective_end);
  Date effective_end;
  if (hasEffectiveEnd) effective_end = hasEffectiveEnd->getDate();
  const DataValue* hasPriority = d.get(Tags::priority);
  int priority = 1;
  if (hasPriority) priority = hasPriority->getInt();
  const DataValue* hasName = d.get(Tags::name);
  string name;
  if (hasName) name = hasName->getString();
  Resource::skilllist::const_iterator s = res->getSkills();
  while (ResourceSkill* i = s.next()) {
    if (i->getSkill() != skill) continue;
    if (hasEffectiveStart && i->getEffectiveStart() != effective_start)
      continue;
    if (hasEffectiveEnd && i->getEffectiveEnd() != effective_end) continue;
    if (hasPriority && i->getPriority() != priority) continue;
    if (hasName && i->getName() != name) continue;
    return const_cast<ResourceSkill*>(&*i);
  }
  return nullptr;
}

}  // namespace frepple
