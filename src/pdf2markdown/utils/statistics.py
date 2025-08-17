"""Statistics tracking for PDF to Markdown conversion."""

import logging
import time
from dataclasses import dataclass, field

from rich.console import Console

logger = logging.getLogger(__name__)


@dataclass
class PageStatistics:
    """Statistics for a single page."""

    page_number: int
    parsing_start: float | None = None
    parsing_end: float | None = None
    conversion_start: float | None = None
    conversion_end: float | None = None
    validation_corrections: int = 0
    validation_issues_found: int = 0
    validation_issues_resolved: int = 0

    @property
    def parsing_duration(self) -> float | None:
        """Get parsing duration in seconds."""
        if self.parsing_start and self.parsing_end:
            return self.parsing_end - self.parsing_start
        return None

    @property
    def conversion_duration(self) -> float | None:
        """Get conversion duration in seconds."""
        if self.conversion_start and self.conversion_end:
            return self.conversion_end - self.conversion_start
        return None

    @property
    def total_duration(self) -> float | None:
        """Get total processing duration in seconds."""
        parsing = self.parsing_duration or 0
        conversion = self.conversion_duration or 0
        return parsing + conversion if (parsing or conversion) else None


@dataclass
class StatisticsTracker:
    """Tracks statistics for the entire conversion process."""

    # Overall timing
    process_start: float | None = None
    process_end: float | None = None

    # Document-level stats
    total_pages: int = 0
    pages_processed: int = 0
    pages_failed: int = 0

    # Page-level statistics
    page_stats: dict[int, PageStatistics] = field(default_factory=dict)

    # Parsing phase (PDF to images)
    parsing_start: float | None = None
    parsing_end: float | None = None

    # Conversion phase (images to markdown)
    conversion_start: float | None = None
    conversion_end: float | None = None

    def start_process(self) -> None:
        """Mark the start of the entire process."""
        self.process_start = time.time()
        logger.debug("Process started")

    def end_process(self) -> None:
        """Mark the end of the entire process."""
        self.process_end = time.time()
        logger.debug("Process ended")

    def start_parsing(self) -> None:
        """Mark the start of document parsing phase."""
        self.parsing_start = time.time()
        logger.debug("Document parsing started")

    def end_parsing(self) -> None:
        """Mark the end of document parsing phase."""
        self.parsing_end = time.time()
        logger.debug("Document parsing ended")

    def start_conversion(self) -> None:
        """Mark the start of conversion phase."""
        self.conversion_start = time.time()
        logger.debug("Conversion to markdown started")

    def end_conversion(self) -> None:
        """Mark the end of conversion phase."""
        self.conversion_end = time.time()
        logger.debug("Conversion to markdown ended")

    def start_page_parsing(self, page_number: int) -> None:
        """Mark the start of parsing for a specific page."""
        if page_number not in self.page_stats:
            self.page_stats[page_number] = PageStatistics(page_number=page_number)
        self.page_stats[page_number].parsing_start = time.time()
        logger.debug(f"Page {page_number} parsing started")

    def end_page_parsing(self, page_number: int) -> None:
        """Mark the end of parsing for a specific page."""
        if page_number in self.page_stats:
            self.page_stats[page_number].parsing_end = time.time()
            logger.debug(f"Page {page_number} parsing ended")

    def start_page_conversion(self, page_number: int) -> None:
        """Mark the start of conversion for a specific page."""
        if page_number not in self.page_stats:
            self.page_stats[page_number] = PageStatistics(page_number=page_number)
        self.page_stats[page_number].conversion_start = time.time()
        logger.debug(f"Page {page_number} conversion started")

    def end_page_conversion(self, page_number: int) -> None:
        """Mark the end of conversion for a specific page."""
        if page_number in self.page_stats:
            self.page_stats[page_number].conversion_end = time.time()
            self.pages_processed += 1
            logger.debug(f"Page {page_number} conversion ended")

    def record_page_failure(self, page_number: int) -> None:
        """Record that a page failed to process."""
        self.pages_failed += 1
        logger.debug(f"Page {page_number} failed")

    def record_validation_stats(
        self,
        page_number: int,
        corrections: int = 0,
        issues_found: int = 0,
        issues_resolved: int = 0,
    ) -> None:
        """Record validation statistics for a page."""
        if page_number not in self.page_stats:
            self.page_stats[page_number] = PageStatistics(page_number=page_number)

        stats = self.page_stats[page_number]
        stats.validation_corrections = corrections
        stats.validation_issues_found = issues_found
        stats.validation_issues_resolved = issues_resolved

    @property
    def total_duration(self) -> float | None:
        """Get total process duration in seconds."""
        if self.process_start and self.process_end:
            return self.process_end - self.process_start
        elif self.process_start:
            return time.time() - self.process_start
        return None

    @property
    def parsing_duration(self) -> float | None:
        """Get total parsing duration in seconds."""
        if self.parsing_start and self.parsing_end:
            return self.parsing_end - self.parsing_start
        return None

    @property
    def conversion_duration(self) -> float | None:
        """Get total conversion duration in seconds."""
        if self.conversion_start and self.conversion_end:
            return self.conversion_end - self.conversion_start
        return None

    @property
    def average_page_parsing_time(self) -> float | None:
        """Get average time to parse a page."""
        times = [
            s.parsing_duration for s in self.page_stats.values() if s.parsing_duration is not None
        ]
        return sum(times) / len(times) if times else None

    @property
    def average_page_conversion_time(self) -> float | None:
        """Get average time to convert a page."""
        times = [
            s.conversion_duration
            for s in self.page_stats.values()
            if s.conversion_duration is not None
        ]
        return sum(times) / len(times) if times else None

    @property
    def total_validation_corrections(self) -> int:
        """Get total number of validation corrections performed."""
        return sum(s.validation_corrections for s in self.page_stats.values())

    @property
    def total_validation_issues(self) -> int:
        """Get total number of validation issues found."""
        return sum(s.validation_issues_found for s in self.page_stats.values())

    @property
    def total_validation_resolved(self) -> int:
        """Get total number of validation issues resolved."""
        return sum(s.validation_issues_resolved for s in self.page_stats.values())

    def format_duration(self, seconds: float | None) -> str:
        """Format duration in human-readable format."""
        if seconds is None:
            return "N/A"

        if seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"

    def generate_report(self) -> str:
        """Generate a statistics report."""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("CONVERSION STATISTICS REPORT")
        lines.append("=" * 60)

        # Overall statistics
        lines.append("\n📊 Overall Statistics:")
        lines.append(f"  Total Pages: {self.total_pages}")
        lines.append(f"  Pages Processed: {self.pages_processed}")
        if self.pages_failed > 0:
            lines.append(f"  Pages Failed: {self.pages_failed}")
        lines.append(f"  Total Duration: {self.format_duration(self.total_duration)}")

        # Parsing phase statistics
        if self.parsing_duration:
            lines.append("\n📄 PDF Parsing Phase (PDF → Images):")
            lines.append(f"  Total Time: {self.format_duration(self.parsing_duration)}")
            if self.total_pages > 0:
                avg_time = self.parsing_duration / self.total_pages
                lines.append(f"  Average per Page: {self.format_duration(avg_time)}")
                pages_per_sec = (
                    self.total_pages / self.parsing_duration if self.parsing_duration > 0 else 0
                )
                lines.append(f"  Throughput: {pages_per_sec:.2f} pages/second")

        # Conversion phase statistics
        if self.conversion_duration:
            lines.append("\n🔄 Conversion Phase (Images → Markdown):")
            lines.append(f"  Total Time: {self.format_duration(self.conversion_duration)}")
            if self.pages_processed > 0:
                avg_time = self.conversion_duration / self.pages_processed
                lines.append(f"  Average per Page: {self.format_duration(avg_time)}")
                pages_per_sec = (
                    self.pages_processed / self.conversion_duration
                    if self.conversion_duration > 0
                    else 0
                )
                lines.append(f"  Throughput: {pages_per_sec:.2f} pages/second")

        # Validation statistics
        if self.total_validation_corrections > 0 or self.total_validation_issues > 0:
            lines.append("\n✅ Validation Statistics:")
            lines.append(f"  Issues Found: {self.total_validation_issues}")
            lines.append(f"  Issues Resolved: {self.total_validation_resolved}")
            lines.append(f"  Correction Attempts: {self.total_validation_corrections}")
            if self.total_validation_issues > 0:
                resolution_rate = (
                    self.total_validation_resolved / self.total_validation_issues
                ) * 100
                lines.append(f"  Resolution Rate: {resolution_rate:.1f}%")

        # Performance summary
        if self.total_duration and self.total_pages > 0:
            lines.append("\n⚡ Performance Summary:")
            overall_pages_per_min = (self.total_pages / self.total_duration) * 60
            lines.append(f"  Overall Throughput: {overall_pages_per_min:.1f} pages/minute")

            if self.parsing_duration and self.conversion_duration:
                parsing_pct = (self.parsing_duration / self.total_duration) * 100
                conversion_pct = (self.conversion_duration / self.total_duration) * 100
                lines.append("  Time Distribution:")
                lines.append(f"    - Parsing: {parsing_pct:.1f}%")
                lines.append(f"    - Conversion: {conversion_pct:.1f}%")
                other_pct = 100 - parsing_pct - conversion_pct
                if other_pct > 1:
                    lines.append(f"    - Other: {other_pct:.1f}%")

        lines.append("=" * 60)
        return "\n".join(lines)

    def print_report(self, console: Console | None = None) -> None:
        """Print the statistics report to console."""
        report = self.generate_report()

        if console:
            console.print(report)
        else:
            print(report)

    def get_summary_dict(self) -> dict:
        """Get summary statistics as a dictionary."""
        return {
            "total_pages": self.total_pages,
            "pages_processed": self.pages_processed,
            "pages_failed": self.pages_failed,
            "total_duration": self.total_duration,
            "parsing_duration": self.parsing_duration,
            "conversion_duration": self.conversion_duration,
            "avg_parsing_time": self.average_page_parsing_time,
            "avg_conversion_time": self.average_page_conversion_time,
            "validation_corrections": self.total_validation_corrections,
            "validation_issues_found": self.total_validation_issues,
            "validation_issues_resolved": self.total_validation_resolved,
        }


# Global statistics tracker instance
_global_stats: StatisticsTracker | None = None


def get_statistics_tracker() -> StatisticsTracker:
    """Get or create the global statistics tracker."""
    global _global_stats
    if _global_stats is None:
        _global_stats = StatisticsTracker()
    return _global_stats


def reset_statistics() -> None:
    """Reset the global statistics tracker."""
    global _global_stats
    _global_stats = StatisticsTracker()


def get_statistics() -> StatisticsTracker | None:
    """Get the current statistics tracker if it exists."""
    return _global_stats
