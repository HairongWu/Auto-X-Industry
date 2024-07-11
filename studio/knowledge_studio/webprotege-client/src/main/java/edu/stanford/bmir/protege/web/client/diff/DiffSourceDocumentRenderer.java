package edu.stanford.bmir.protege.web.client.diff;

import com.google.gwt.safehtml.shared.SafeHtml;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 29/01/15
 */
public interface DiffSourceDocumentRenderer<D> {

    SafeHtml renderSourceDocument(D document);
}
