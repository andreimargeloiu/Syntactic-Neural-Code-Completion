

package uk.ac.cam.acr31.features.javac.testing;

import javax.annotation.processing.Generated;

@Generated("com.google.auto.value.processor.AutoValueProcessor")
final class AutoValue_SourceSpan extends SourceSpan {

  private final int start;

  private final int end;

  AutoValue_SourceSpan(
      int start,
      int end) {
    this.start = start;
    this.end = end;
  }

  @Override
  public int start() {
    return start;
  }

  @Override
  public int end() {
    return end;
  }

  @Override
  public String toString() {
    return "SourceSpan{"
         + "start=" + start + ", "
         + "end=" + end
        + "}";
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (o instanceof SourceSpan) {
      SourceSpan that = (SourceSpan) o;
      return (this.start == that.start())
           && (this.end == that.end());
    }
    return false;
  }

  @Override
  public int hashCode() {
    int h$ = 1;
    h$ *= 1000003;
    h$ ^= start;
    h$ *= 1000003;
    h$ ^= end;
    return h$;
  }

}
