package edu.stanford.bmir.protege.web.server.mansyntax.render;

import org.semanticweb.owlapi.model.OWLObject;

import java.util.List;

/**
 * @author Matthew Horridge, Stanford University, Bio-Medical Informatics Research Group, Date: 24/02/2014
 */
public abstract class AbstractSectionRenderer<E extends OWLObject, I, R> implements FrameSectionRenderer<E, I, R> {

    @Override
    public String getSeparatorAfter(int renderableIndex, List<R> renderables) {
        return ", ";
    }
}
