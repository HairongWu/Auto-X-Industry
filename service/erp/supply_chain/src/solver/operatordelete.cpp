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
#include "frepple/solver.h"

namespace frepple {

const MetaClass* OperatorDelete::metadata;

int OperatorDelete::initialize() {
  // Initialize the metadata
  metadata = MetaClass::registerClass<OperatorDelete>(
      "solver", "solver_delete", Object::create<OperatorDelete>);

  // Initialize the Python class
  PythonType& x = FreppleClass<OperatorDelete, Solver>::getPythonType();
  x.setName("solver_delete");
  x.setDoc("frePPLe solver_delete");
  x.supportgetattro();
  x.supportsetattro();
  x.supportcreate(create);
  x.addMethod("solve", solve, METH_VARARGS, "run the solver");
  metadata->setPythonClass(x);
  return x.typeReady();
}

PyObject* OperatorDelete::create(PyTypeObject* pytype, PyObject* args,
                                 PyObject* kwds) {
  try {
    // Create the solver
    OperatorDelete* s = new OperatorDelete();

    // Iterate over extra keywords, and set attributes.   @todo move this
    // responsibility to the readers...
    if (kwds) {
      PyObject *key, *value;
      Py_ssize_t pos = 0;
      while (PyDict_Next(kwds, &pos, &key, &value)) {
        PythonData field(value);
        PyObject* key_utf8 = PyUnicode_AsUTF8String(key);
        DataKeyword attr(PyBytes_AsString(key_utf8));
        Py_DECREF(key_utf8);
        const MetaFieldBase* fmeta =
            OperatorDelete::metadata->findField(attr.getHash());
        if (!fmeta) fmeta = Solver::metadata->findField(attr.getHash());
        if (fmeta)
          // Update the attribute
          fmeta->setField(s, field);
        else
          s->setProperty(attr.getName(), value);
        ;
      };
    }

    // Return the object
    Py_INCREF(s);
    return static_cast<PyObject*>(s);
  } catch (...) {
    PythonType::evalException();
    return nullptr;
  }
}

void OperatorDelete::solve(void* v) {
  // Loop over all buffers Push to stack, in order of level TODO
  // Clean up all buffers in the list
  while (!buffersToScan.empty()) {
    Buffer* curbuf = buffersToScan.back();
    buffersToScan.pop_back();
    solve(curbuf);
  }
}

void OperatorDelete::solve(OperationPlan* o, void* v) {
  if (!o) return;  // Null argument passed

  // Mark all buffers.
  // The batching solver doesn't like that we push both consumers and
  // producers, but ideally we would pass true for both arguments.
  pushBuffers(o, true, false);

  // Delete the operationplan
  if (o->getProposed()) {
    if (cmds)
      cmds->add(new CommandDeleteOperationPlan(o));
    else
      delete o;
  }

  // Propagate to all upstream buffers
  while (!buffersToScan.empty()) {
    Buffer* curbuf = buffersToScan.back();
    buffersToScan.pop_back();
    solve(curbuf);
  }
}

void OperatorDelete::solve(const Resource* r, void* v) {
  if (getLogLevel() > 0) logger << "Scanning " << r << " for excess" << endl;

  // Loop over all operationplans on the resource
  for (Resource::loadplanlist::const_iterator i = r->getLoadPlans().begin();
       i != r->getLoadPlans().end(); ++i) {
    if (i->getEventType() == 1)
      // Add all buffers into which material is produced to the stack
      pushBuffers(i->getOperationPlan(), false, true);
  }

  // Process all buffers found, and their upstream colleagues
  while (!buffersToScan.empty()) {
    Buffer* curbuf = buffersToScan.back();
    buffersToScan.pop_back();
    solve(curbuf);
  }
}

void OperatorDelete::solve(const Demand* d, void* v) {
  if (getLogLevel() > 1) logger << "Scanning " << d << " for excess" << endl;

  // Delete all delivery operationplans.
  // Note that an extra loop is used to assure that our iterator doesn't get
  // invalidated during the deletion.
  while (true) {
    // Find a candidate operationplan to delete
    OperationPlan* candidate = nullptr;
    const Demand::OperationPlanList& deli = d->getDelivery();
    for (Demand::OperationPlanList::const_iterator i = deli.begin();
         i != deli.end(); ++i)
      if ((*i)->getProposed()) {
        candidate = *i;
        break;
      }
    if (!candidate) break;

    // Push the buffer on the stack in which the deletion creates excess
    // inventory
    pushBuffers(candidate, true, false);

    // Delete only the delivery, immediately or through a delete command
    if (cmds)
      cmds->add(new CommandDeleteOperationPlan(candidate));
    else
      delete candidate;
  }

  // Propagate to all upstream buffers
  while (!buffersToScan.empty()) {
    Buffer* curbuf = buffersToScan.back();
    buffersToScan.pop_back();
    solve(curbuf);
  }
}

void OperatorDelete::pushBuffers(OperationPlan* o, bool consuming,
                                 bool producing) {
  // Loop over all flowplans
  for (OperationPlan::FlowPlanIterator i = o->beginFlowPlans();
       i != o->endFlowPlans(); ++i) {
    // Skip flowplans we're not interested in
    if (!(consuming && i->getQuantity() < 0) &&
        !(producing && i->getQuantity() > 0))
      continue;

    // Check if the buffer is already found on the stack
    bool found = false;
    for (vector<Buffer*>::const_reverse_iterator j = buffersToScan.rbegin();
         j != buffersToScan.rend(); ++j) {
      if (*j == i->getBuffer()) {
        found = true;
        break;
      }
    }

    // Add the buffer to the stack
    if (!found) buffersToScan.push_back(const_cast<Buffer*>(i->getBuffer()));
  }

  // Recursive call for all suboperationplans
  for (OperationPlan::iterator subopplan(o); subopplan != OperationPlan::end();
       ++subopplan)
    pushBuffers(&*subopplan, consuming, producing);
}

void OperatorDelete::solve(const Buffer* b, void* v) {
  if (getLogLevel() > 1) logger << "Scanning buffer " << b << endl;

  Buffer::flowplanlist::const_iterator fiter = b->getFlowPlans().begin();
  Buffer::flowplanlist::const_iterator fend = b->getFlowPlans().end();
  if (fiter == fend) return;  // There isn't a single flowplan in the buffer

  // STEP 1: Remove shortages from the buffer
  // Delete the earliest unlocked consumer(s) after the start of a material
  // shortage.
  if (getConstrained()) {
    double unresolvable = 0.0;

    while (fiter != fend) {
      if (fiter->getQuantity() >= 0 ||
          !(fiter->getOnhand() < -ROUNDING_ERROR + unresolvable &&
            fiter->isLastOnDate())) {
        // Not a consumer or no shortage start
        ++fiter;
        continue;
      }

      // Recurse backward to find consumers we can resize
      double cur_shortage = fiter->getOnhand() + unresolvable;
      Buffer::flowplanlist::const_iterator fiter2 = fiter;
      OperationPlan* curopplan = fiter->getOperationPlan();
      do
        ++fiter;  // increment to an event after the shortage start, because the
                  // iterator can get invalidated in the next loop
      while (fiter != fend && curopplan &&
             fiter->getOperationPlan() ==
                 curopplan);  // A loop is required to handle transfer batches
      while (cur_shortage <= -ROUNDING_ERROR && fiter2 != fend) {
        if (fiter2->getQuantity() >= 0 || fiter2->getEventType() != 1) {
          // Not a consuming flowplan
          --fiter2;
          continue;
        }
        FlowPlan* fp =
            const_cast<FlowPlan*>(static_cast<const FlowPlan*>(&*fiter2));
        if (!fp->getOperationPlan()->getProposed()) {
          // This consumer is locked
          --fiter2;
          continue;
        }

        // Decrement the iterator here, because it can get invalidated later on
        while (fiter2 != fend && fiter2->getEventType() == 1 &&
               fiter2->getOperationPlan()->getTopOwner() ==
                   fp->getOperationPlan()->getTopOwner())
          --fiter2;

        // Resize or delete the candidate operationplan
        double oldsize_flowplan = fp->getQuantity();
        double newsize_opplan;
        double newsize_flowplan;
        if (cur_shortage < fp->getQuantity() + ROUNDING_ERROR) {
          // Completely delete the consumer
          newsize_opplan = newsize_flowplan = 0.0;
        } else {
          // Resize the consumer
          auto tmp = fp->setQuantity(fp->getQuantity() - cur_shortage, true,
                                     false, true, 0);
          newsize_flowplan = tmp.first;
          newsize_opplan = tmp.second;
        }
        if (newsize_flowplan > -ROUNDING_ERROR) {
          // The complete operationplan is shortage.
          cur_shortage -= oldsize_flowplan;
          // Add downstream buffers to the stack
          pushBuffers(fp->getOperationPlan(), false, true);
          // Log message
          if (getLogLevel() > 0)
            logger << "Removing shortage operationplan: "
                   << fp->getOperationPlan() << endl;
          // Delete operationplan
          if (cmds)
            cmds->add(new CommandDeleteOperationPlan(fp->getOperationPlan()));
          else
            delete fp->getOperationPlan();
        } else {
          // Reduce the operationplan
          // Add downstream buffers to the stack
          pushBuffers(fp->getOperationPlan(), false, true);
          // Reduce the shortage
          cur_shortage -= oldsize_flowplan - newsize_flowplan;
          if (getLogLevel() > 0)
            logger << "Resizing shortage operationplan to " << newsize_opplan
                   << ": " << fp->getOperationPlan() << endl;
          // Resize operationplan
          if (cmds)
            // TODO Incorrect - need to resize the flowplan intead of the the
            // operationplan!
            cmds->add(new CommandMoveOperationPlan(
                fp->getOperationPlan(), fp->getOperationPlan()->getStart(),
                Date::infinitePast, newsize_opplan));
          else
            fp->getOperationPlan()->setQuantity(newsize_opplan);
        }
      }

      // Damn... We can't resolve it
      if (fiter2 == fend && cur_shortage <= -ROUNDING_ERROR) {
        unresolvable += cur_shortage;
        if (getLogLevel() > 0)
          logger << "Can't resolve shortage problem in buffer " << b << endl;
      }
    }
  }

  // STEP 2: Remove excess inventory at the end of the planning horizon.
  // Delete the earliest unlocked producer(s) that leave(s) excess at any later
  // point in the horizon.
  fiter = b->getFlowPlans().rbegin();
  if (fiter == fend) return;
  double excess =
      fiter->getOnhand() - max(max(fiter->getMin(), fiter->getMax()), 0.0);
  if (excess > ROUNDING_ERROR) {
    fiter = b->getFlowPlans().begin();
    while (excess > ROUNDING_ERROR && fiter != fend) {
      if (fiter->getQuantity() <= 0) {
        // Not a producer
        ++fiter;
        continue;
      }
      FlowPlan* fp = nullptr;
      if (fiter->getEventType() == 1)
        fp = const_cast<FlowPlan*>(static_cast<const FlowPlan*>(&*fiter));
      if (!fp || !fp->getOperationPlan()->getProposed() ||
          fp->getOperationPlan()->getDemand() ||
          (fp->getOperationPlan()->getOwner() &&
           fp->getOperationPlan()->getOwner()->getDemand()) ||
          fp->getFlow()->hasType<FlowTransferBatch>()) {
        // It's locked or a delivery operationplan
        // TODO we currently also exclude transferbatch producers, which isn't
        // really correct
        ++fiter;
        continue;
      }

      // Compute the excess quantity
      double cur_excess = b->getFlowPlans().getExcess(&*fiter);
      if (fp)
        for (auto flpln_on_opplan = fp->getOperationPlan()->beginFlowPlans();
             flpln_on_opplan != fp->getOperationPlan()->endFlowPlans();
             ++flpln_on_opplan) {
          if (flpln_on_opplan->getQuantity() < ROUNDING_ERROR ||
              flpln_on_opplan->getBuffer() == b ||
              !flpln_on_opplan->getFlow()->getQuantity())
            continue;
          auto myexcess =
              (flpln_on_opplan->getBuffer()->getFlowPlans().getExcess(
                   &*flpln_on_opplan) -
               flpln_on_opplan->getFlow()->getQuantityFixed()) *
              fp->getFlow()->getQuantity() /
              flpln_on_opplan->getFlow()->getQuantity();
          if (myexcess >= 0.0 && myexcess < cur_excess) cur_excess = myexcess;
        }
      if (cur_excess < ROUNDING_ERROR) {
        // It doesn't produce excess
        ++fiter;
        continue;
      }

      // Increment the iterator here, because it can get invalidated later on
      while (fiter != fend && fiter->getEventType() == 1 &&
             fiter->getOperationPlan()->getTopOwner() ==
                 fp->getOperationPlan()->getTopOwner())
        ++fiter;

      double newsize_opplan;
      double newsize_flowplan;

      auto topopplan = fp->getOperationPlan();
      if (topopplan->getOwner() &&
          topopplan->getOwner()->getOperation()->hasType<OperationRouting>())
        topopplan = topopplan->getOwner();

      if (cur_excess < fp->getFlow()->getQuantityFixed() +
                           topopplan->getOperation()->getSizeMultiple() *
                               fp->getFlow()->getQuantity())
        // This excess is unavoidable...
        continue;
      else if (cur_excess >= fp->getQuantity() - ROUNDING_ERROR) {
        // Completely delete the producer
        newsize_opplan = newsize_flowplan = 0.0;
      } else {
        // Resize the producer
        // We need to keep the operationplan start date constant during the
        // resize to avoid that a capacity consumption from a bucketized
        // resource moves to a different bucket.
        auto tmp = fp->setQuantity(fp->getQuantity() - cur_excess, false, false,
                                   true, 1);
        newsize_flowplan = tmp.first;
        newsize_opplan = tmp.second;
      }
      if (newsize_flowplan < ROUNDING_ERROR) {
        // The complete operationplan is excess.
        // Reduce the excess
        excess -= fp->getQuantity();
        // Add upstream buffers to the stack
        pushBuffers(fp->getOperationPlan(), true, false);
        // Log message
        if (getLogLevel() > 0)
          logger << "Removing excess operationplan: " << topopplan << endl;
        // Delete operationplan
        if (cmds)
          cmds->add(new CommandDeleteOperationPlan(topopplan));
        else
          delete fp->getOperationPlan();
      } else if (newsize_opplan <
                 fp->getOperationPlan()->getQuantity() - ROUNDING_ERROR) {
        // Reduce the operationplan
        auto delta = fp->getQuantity() - newsize_flowplan;
        if (delta > ROUNDING_ERROR) {
          // Add upstream buffers to the stack
          pushBuffers(fp->getOperationPlan(), true, false);
          // Reduce the excess
          excess -= fp->getQuantity() - newsize_flowplan;
          if (getLogLevel() > 0)
            logger << "Resizing excess operationplan to " << newsize_opplan
                   << ": " << topopplan << endl;
          // Resize operationplan
          if (cmds)
            // TODO Incorrect - need to resize the flowplan intead of the the
            // operationplan!
            cmds->add(new CommandMoveOperationPlan(
                topopplan, Date::infinitePast, topopplan->getEnd(),
                newsize_opplan));
          else
            topopplan->setQuantity(newsize_opplan);
        }
      }
    }
  }
}

PyObject* OperatorDelete::solve(PyObject* self, PyObject* args) {
  // Parse the argument
  PyObject* obj = nullptr;
  short objtype = 0;
  if (args && !PyArg_ParseTuple(args, "|O:solve", &obj)) return nullptr;
  if (obj) {
    if (PyObject_TypeCheck(obj, Demand::metadata->pythonClass))
      objtype = 1;
    else if (PyObject_TypeCheck(obj, Buffer::metadata->pythonClass))
      objtype = 2;
    else if (PyObject_TypeCheck(obj, Resource::metadata->pythonClass))
      objtype = 3;
    else if (PyObject_TypeCheck(obj, OperationPlan::metadata->pythonClass))
      objtype = 4;
    else {
      PyErr_SetString(PythonDataException,
                      "solve(d) argument must be a demand, buffer, resource or "
                      "operationplan");
      return nullptr;
    }
  }

  // Free Python interpreter for other threads
  Py_BEGIN_ALLOW_THREADS;
  try {
    OperatorDelete* sol = static_cast<OperatorDelete*>(self);
    switch (objtype) {
      case 0:
        // Delete all excess
        sol->solve();
        break;
      case 1:
        // Delete upstream of a single demand
        sol->solve(static_cast<Demand*>(obj));
        break;
      case 2:
        // Delete upstream of a single buffer
        sol->solve(static_cast<Buffer*>(obj));
        break;
      case 3:
        // Delete upstream of a single resource
        sol->solve(static_cast<Resource*>(obj));
        break;
      case 4:
        // Delete an operationplan
        sol->solve(static_cast<OperationPlan*>(obj));
    }
  } catch (...) {
    Py_BLOCK_THREADS;
    PythonType::evalException();
    return nullptr;
  }
  // Reclaim Python interpreter
  Py_END_ALLOW_THREADS;
  return Py_BuildValue("");
}

}  // namespace frepple
