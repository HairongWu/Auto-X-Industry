package edu.stanford.bmir.protege.web.server.mansyntax.render;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import edu.stanford.bmir.protege.web.shared.inject.ProjectSingleton;
import org.obolibrary.obo2owl.Obo2OWLConstants;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.vocab.DublinCoreVocabulary;
import org.semanticweb.owlapi.vocab.OWLRDFVocabulary;
import org.semanticweb.owlapi.vocab.SKOSVocabulary;

import java.util.Map;

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * @author Matthew Horridge, Stanford University, Bio-Medical Informatics Research Group, Date: 03/10/2014
 *
 * Provides a predefined ordinal of IRIs for tasks such as ordering.
 */
@ProjectSingleton
public class IRIOrdinalProvider {

    private final ImmutableMap<IRI, Integer> indexMap;

    private static final ImmutableList<IRI> DEFAULT_ANNOTATION_PROPERTY_ORDERING = ImmutableList.<IRI>builder().add(
                // Labels
                OWLRDFVocabulary.RDFS_LABEL.getIRI(),
                SKOSVocabulary.PREFLABEL.getIRI(),
                DublinCoreVocabulary.TITLE.getIRI(),

                // OBO stuff that's important
                // Id.  I can't find a constant for this.
                IRI.create("http://www.geneontology.org/formats/oboInOwl#id"),

                // Replaced by
                Obo2OWLConstants.Obo2OWLVocabulary.IRI_IAO_0100001.getIRI(),

                Obo2OWLConstants.Obo2OWLVocabulary.hasAlternativeId.getIRI(),

                // Definitions
                Obo2OWLConstants.Obo2OWLVocabulary.IRI_IAO_0000115.getIRI(),

                Obo2OWLConstants.Obo2OWLVocabulary.IRI_OIO_hasOboNamespace.getIRI(),

                // Other definitions
                SKOSVocabulary.DEFINITION.getIRI(),
                DublinCoreVocabulary.DESCRIPTION.getIRI(),
                SKOSVocabulary.NOTE.getIRI(),

                OWLRDFVocabulary.RDFS_COMMENT.getIRI(),

                // Other labels
                SKOSVocabulary.ALTLABEL.getIRI(),

                OWLRDFVocabulary.RDFS_SEE_ALSO.getIRI(),
                OWLRDFVocabulary.RDFS_IS_DEFINED_BY.getIRI(),

                // Important OBO annotations.  Ordering derived from documents shared by Chris M.
                Obo2OWLConstants.Obo2OWLVocabulary.IRI_OIO_hasExactSynonym.getIRI(),
                Obo2OWLConstants.Obo2OWLVocabulary.IRI_OIO_hasRelatedSynonym.getIRI(),
                Obo2OWLConstants.Obo2OWLVocabulary.IRI_OIO_hasBroadSynonym.getIRI(),
                Obo2OWLConstants.Obo2OWLVocabulary.IRI_OIO_hasNarrowSynonym.getIRI(),

                Obo2OWLConstants.Obo2OWLVocabulary.IRI_OIO_hasDbXref.getIRI(),
                Obo2OWLConstants.Obo2OWLVocabulary.IRI_OIO_Subset.getIRI()
        ).build();

    private static final IRIOrdinalProvider INDEX_WITH_DEFAULT_ORDERING =
            new IRIOrdinalProvider(DEFAULT_ANNOTATION_PROPERTY_ORDERING);

    /**
     * Constructs an index provider for the specified list of IRIs.
     * @param iris The list of IRIs.  Not {@code null}.
     * @throws NullPointerException if the list is {@code null}.
     */
    public IRIOrdinalProvider(ImmutableList<IRI> iris) {
        checkNotNull(iris);
        Map<IRI, Integer> map = Maps.newHashMap();
        for(IRI iri : iris) {
            map.put(iri, map.size());
        }
        indexMap = ImmutableMap.copyOf(map);
    }

    /**
     * Gets the index of the specified IRI.
     * @param iri The IRI whose index should be retrieved.
     * @return A non-negative integer that represents the IRI index.  If the IRI was not specified in the IRI list in
     * the constructor for this object then the returned index will be the size of the specified list.
     */
    public int getIndex(IRI iri) {
        Integer index = indexMap.get(iri);
        if(index == null) {
            return indexMap.size();
        }
        else {
            return index;
        }
    }

    /**
     * Gets the index based on a predefined default ordering for annotation property IRIs.
     * @return The IRIOrdinalProvider for the default annotation property IRI ordering.  Not {@code null}.
     */
    public static IRIOrdinalProvider withDefaultAnnotationPropertyOrdering() {
        return INDEX_WITH_DEFAULT_ORDERING;
    }



}
